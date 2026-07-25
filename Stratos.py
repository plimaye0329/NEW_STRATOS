#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   STRATOS — Signal Time-domain Research and Analysis Tools for ObservationS ║
╚══════════════════════════════════════════════════════════════════════════════╝

A modular pipeline for pulsar single-pulse analysis.

COMMANDS
--------
  read          Read a filterbank file, dedisperse, and save timeseries
  clean         Clean RFI channels and produce a normalised timeseries CSV
  extract       Extract individual pulses from a timeseries
  fit           Multi-Gaussian fit all pulse files with chi-square minimisation
  fluence       Measure fluence and width for all pulses from fit results
  energy        Convert fluence → energy (optionally with w50 / SNR path)
  run           Run the full pipeline end-to-end (read → clean → extract → fit → fluence)

Run  stratos.py <command> --help  for per-command options.

EXAMPLES
--------
  # Full pipeline
  python stratos.py run --fil obs.fil --dm 26.8 --block 100000 \\
      --bad-channels 0 50 200 300 --output-dir ./results

  # Individual steps
  python stratos.py read  --fil obs.fil --dm 26.8 --block 100000 -o ts.txt
  python stratos.py clean --ts ts.txt --bad-channels 0 50 200 300 -o cleaned.txt
  python stratos.py extract --ts cleaned.txt --distance 160 --prominence 8 --width 200
  python stratos.py fit    --pulse-dir ./pulses --samp-rate 0.025
  python stratos.py fluence --fit-dir ./pulses --samp-rate 0.025 --sefd 17 --bw 400e6
"""

import argparse
import os
import sys
import glob
import warnings
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

# ── Optional heavy deps (only needed for read/clean) ──────────────────────────
try:
    import sigpyproc.readers as readers
    HAS_SIGPYPROC = True
except ImportError:
    HAS_SIGPYPROC = False

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
  _____ _____ ____      _  _____ ___  ____
 / ___|_   _|  _ \    / \|_   _/ _ \/ ___|
 \___ \ | | | |_) |  / _ \ | || | | \___ \
  ___) || | |  _ <  / ___ \| || |_| |___) |
 |____/ |_| |_| \_\/_/   \_\_| \___/|____/

 Signal Time-domain Research and Analysis Tools for ObservationS
"""

# ══════════════════════════════════════════════════════════════════════════════
#  GAUSSIAN UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2.0 * stddev ** 2))


def multi_gaussian(x, *params):
    """Sum of N Gaussians; params = [amp0, mu0, sig0,  amp1, mu1, sig1, ...]"""
    result = np.zeros_like(x, dtype=np.float64)
    n = len(params) // 3
    for i in range(n):
        result += gaussian(x, params[i*3], params[i*3+1], params[i*3+2])
    return result


def fwhm_intercepts(mu, sigma):
    """Return (left, right) FWHM intercepts for a single Gaussian."""
    half_width = abs(sigma) * np.sqrt(2.0 * np.log(2.0))
    return mu - half_width, mu + half_width


def reduced_chi_square(observed, model, n_params, noise_std):
    """Reduced chi-square statistic."""
    if noise_std <= 0:
        noise_std = 1e-10
    dof = max(len(observed) - n_params, 1)
    return np.sum(((observed - model) / noise_std) ** 2) / dof


# ══════════════════════════════════════════════════════════════════════════════
#  ITERATIVE MULTI-GAUSSIAN FITTER (chi-square minimisation)
# ══════════════════════════════════════════════════════════════════════════════

def fit_multigaussian(
    x, y,
    chi2_threshold=1.5,
    max_gaussians=10,
    min_sigma=0.5,
    sn_threshold=3.0,
    noise_std=None,
):
    """
    Fit an increasing number of Gaussians until reduced chi-square ≤ chi2_threshold.

    Strategy
    --------
    1. Estimate noise from the edge bins.
    2. Start with 0 Gaussians (residual = y).
    3. Each iteration: find the tallest peak in the residual, seed a new
       Gaussian there, refit ALL Gaussians jointly with curve_fit.
    4. Stop when χ²_red ≤ chi2_threshold OR no more significant peaks.

    Returns
    -------
    dict with keys:
        params        : flat list [amp, mu, sig, ...]
        chi2_red      : final reduced chi-square
        n_gaussians   : number of components
        envelope_left : left FWHM intercept of leftmost Gaussian (in x units)
        envelope_right: right FWHM intercept of rightmost Gaussian (in x units)
        envelope_width: envelope_right - envelope_left  (in x units)
    """
    if noise_std is None:
        edge = max(5, len(y) // 10)
        noise_std = np.std(np.concatenate([y[:edge], y[-edge:]]))
        if noise_std <= 0:
            noise_std = 1e-10

    fitted_params = []
    residual      = y.copy()
    chi2_red      = np.inf

    for _ in range(max_gaussians):
        # ── Find tallest peak in current residual ──────────────────────────
        smooth_res = residual.copy()
        peaks, props = find_peaks(smooth_res, height=sn_threshold * noise_std)
        if len(peaks) == 0:
            break

        best_peak    = peaks[np.argmax(smooth_res[peaks])]
        guess_amp    = smooth_res[best_peak]
        guess_mu     = x[best_peak]
        guess_sigma  = max(min_sigma, (x[1] - x[0]) * 2)

        # ── Build initial guess including all previous Gaussians ───────────
        p0 = fitted_params + [guess_amp, guess_mu, guess_sigma]

        # ── Bounds: amplitudes > 0, sigma > min_sigma ─────────────────────
        n_new   = len(p0) // 3
        lb      = [-np.inf, x[0],     min_sigma] * n_new
        ub      = [ np.inf, x[-1], (x[-1]-x[0])] * n_new

        try:
            popt, _ = curve_fit(
                multi_gaussian, x, y,
                p0=p0, bounds=(lb, ub),
                maxfev=10000,
            )
        except (RuntimeError, ValueError):
            break

        model    = multi_gaussian(x, *popt)
        chi2_red = reduced_chi_square(y, model, len(popt), noise_std)
        fitted_params = popt.tolist()
        residual      = y - model

        if chi2_red <= chi2_threshold:
            break

    # ── Compute envelope from FWHM intercepts ─────────────────────────────
    if not fitted_params:
        return {
            "params": [], "chi2_red": chi2_red, "n_gaussians": 0,
            "envelope_left": np.nan, "envelope_right": np.nan,
            "envelope_width": np.nan,
        }

    n  = len(fitted_params) // 3
    intercepts = []
    for i in range(n):
        mu  = fitted_params[i*3 + 1]
        sig = fitted_params[i*3 + 2]
        l, r = fwhm_intercepts(mu, sig)
        intercepts.append((l, r))

    env_left  = min(ic[0] for ic in intercepts)
    env_right = max(ic[1] for ic in intercepts)

    return {
        "params"         : fitted_params,
        "chi2_red"       : chi2_red,
        "n_gaussians"    : n,
        "envelope_left"  : env_left,
        "envelope_right" : env_right,
        "envelope_width" : env_right - env_left,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FLUENCE WITHIN ENVELOPE
# ══════════════════════════════════════════════════════════════════════════════

def measure_fluence_from_envelope(x, y, env_left, env_right, samp_rate_ms,
                                  sefd=17.0, bw_hz=400e6, n_pol=2):
    """
    Integrate S/N within [env_left, env_right] and convert to fluence (Jy·s).

    Parameters
    ----------
    x           : sample-index array
    y           : normalised S/N timeseries
    env_left    : left envelope boundary (in sample-index units matching x)
    env_right   : right envelope boundary
    samp_rate_ms: time per sample in ms
    sefd        : system equivalent flux density in Jy
    bw_hz       : bandwidth in Hz
    n_pol       : number of polarisations

    Returns
    -------
    fluence_jy_s : fluence in Jy·s
    width_ms     : envelope width in ms
    """
    dt_s    = samp_rate_ms * 1e-3
    dt_ms   = samp_rate_ms

    # Clip to array bounds
    i_left  = int(np.clip(np.searchsorted(x, env_left),  0, len(x)-1))
    i_right = int(np.clip(np.searchsorted(x, env_right), 0, len(x)-1))

    if i_right <= i_left:
        return np.nan, np.nan

    segment   = y[i_left:i_right]
    width_ms  = (i_right - i_left) * dt_ms

    # Radiometer equation integration
    # S = SEFD * SNR / sqrt(n_pol * bw * dt)
    # Fluence = sum_i  S_i * dt  =  SEFD * dt / sqrt(n_pol * bw * dt) * sum(SNR_i)
    fluence_jy_s = (sefd / np.sqrt(n_pol * bw_hz * dt_s)) * np.sum(segment) * dt_s

    return fluence_jy_s, width_ms


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEPS  (each callable independently)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. READ ───────────────────────────────────────────────────────────────────

def step_read(filepath, dm, block_size, output='timeseries_raw.txt'):
    """Read a filterbank, dedisperse, and save the raw 2-D array."""
    if not HAS_SIGPYPROC:
        raise ImportError("sigpyproc is required for 'read'. Install it with: pip install sigpyproc")
    fil  = readers.FilReader(filepath)
    data = fil.read_block(1, block_size)
    arr  = data.dedisperse(dm).copy()
    np.savetxt(output, arr)
    print(f"[read]  Saved dedispersed block → {output}  shape={arr.shape}")
    return arr


# ── 2. CLEAN ──────────────────────────────────────────────────────────────────

def step_clean(data_or_file, bad_channel_pairs=None, samp_rate=0.025,
               output='timeseries_cleaned.txt'):
    """
    Mask bad channel ranges, scrunch to timeseries, normalise, save.

    Parameters
    ----------
    data_or_file      : 2-D ndarray OR path to a text file saved by step_read
    bad_channel_pairs : list of (start, end) tuples of channel indices to mask
    samp_rate         : ms per sample
    output            : output filename
    """
    if isinstance(data_or_file, str):
        data = np.loadtxt(data_or_file)
    else:
        data = data_or_file.copy()

    if bad_channel_pairs:
        for start, end in bad_channel_pairs:
            data[start:end, :] = np.nan

    intensity = np.nanmean(data, axis=0)
    time_ms   = np.arange(len(intensity)) * samp_rate

    # Normalise: subtract baseline (first 10 %) and divide by its std
    n_base     = max(50, len(intensity) // 10)
    baseline   = intensity[:n_base]
    intensity  = (intensity - np.nanmean(baseline)) / (np.nanstd(baseline) + 1e-12)

    out = np.column_stack((time_ms, intensity))
    np.savetxt(output, out, header='time_ms\tSNR', delimiter='\t', fmt='%.6f')
    print(f"[clean] Saved normalised timeseries → {output}  n_samples={len(intensity)}")
    return out


# ── 3. EXTRACT ────────────────────────────────────────────────────────────────

def step_extract(ts_file, distance=160, prominence=8, half_width=200,
                 save_dir='./pulses', prefix='pulse'):
    """
    Detect peaks and save ±half_width-sample windows around each.

    Returns list of saved filenames.
    """
    os.makedirs(save_dir, exist_ok=True)
    data      = np.loadtxt(ts_file)
    time_ms   = data[:, 0]
    intensity = data[:, 1]

    peaks, _ = find_peaks(intensity, distance=distance, prominence=prominence)
    saved    = []

    for idx, pk in enumerate(peaks, start=1):
        i0 = max(pk - half_width, 0)
        i1 = min(pk + half_width, len(intensity))
        seg  = intensity[i0:i1]
        xseg = np.arange(len(seg), dtype=float)

        if seg.size == 0:
            continue

        fname = os.path.join(save_dir, f"{prefix}_P{idx:04d}.txt")
        np.savetxt(fname, np.column_stack((xseg, seg)),
                   header='sample\tSNR', delimiter='\t', fmt='%.6f')
        saved.append(fname)

    # Save peak positions
    peak_file = os.path.join(save_dir, f"{prefix}_peaks.txt")
    np.savetxt(peak_file,
               np.column_stack((peaks, time_ms[peaks])),
               header='sample_index\ttime_ms', delimiter='\t', fmt=['%d', '%.6f'])

    print(f"[extract] Found {len(peaks)} peaks → saved {len(saved)} pulses in {save_dir}/")
    return saved


# ── 4. FIT ────────────────────────────────────────────────────────────────────

def step_fit(pulse_dir, samp_rate=0.025, chi2_threshold=1.5,
             max_gaussians=10, sn_threshold=3.0,
             output_dir=None, plot_pdf=None):
    """
    Run iterative multi-Gaussian fitting on every pulse txt file.

    Saves a <pulse>_fit.txt per pulse containing fit parameters and envelope.
    Optionally writes a diagnostic PDF with one page per pulse.

    Returns a list of result dicts.
    """
    pulse_files = sorted(glob.glob(os.path.join(pulse_dir, '*.txt')))
    # Exclude any already-generated files from this step
    pulse_files = [f for f in pulse_files
                   if '_fit.txt' not in f and '_peaks.txt' not in f]

    if not pulse_files:
        print(f"[fit] No pulse txt files found in {pulse_dir}")
        return []

    out_dir = output_dir or pulse_dir
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    pdf = PdfPages(plot_pdf) if plot_pdf else None

    print(f"[fit] Fitting {len(pulse_files)} pulses  "
          f"(chi2_threshold={chi2_threshold}, max_gaussians={max_gaussians})")

    for pulse_file in pulse_files:
        try:
            raw = np.loadtxt(pulse_file)
            if raw.ndim != 2 or raw.shape[1] < 2:
                continue
            x = raw[:, 0]
            y = raw[:, 1]
        except Exception as e:
            print(f"  [fit] Could not load {pulse_file}: {e}")
            continue

        result = fit_multigaussian(
            x, y,
            chi2_threshold=chi2_threshold,
            max_gaussians=max_gaussians,
            sn_threshold=sn_threshold,
        )
        result['file'] = pulse_file
        all_results.append(result)

        # ── Save per-pulse fit summary ────────────────────────────────────
        base    = os.path.splitext(os.path.basename(pulse_file))[0]
        fit_out = os.path.join(out_dir, f"{base}_fit.txt")

        with open(fit_out, 'w') as fh:
            fh.write(f"# Multi-Gaussian fit summary for {base}\n")
            fh.write(f"# chi2_red       = {result['chi2_red']:.4f}\n")
            fh.write(f"# n_gaussians    = {result['n_gaussians']}\n")
            fh.write(f"# envelope_left  = {result['envelope_left']:.4f}  (samples)\n")
            fh.write(f"# envelope_right = {result['envelope_right']:.4f}  (samples)\n")
            fh.write(f"# envelope_width_ms = {result['envelope_width'] * samp_rate:.4f}\n")
            fh.write("# amp\tmu\tsigma\n")
            params = result['params']
            for i in range(result['n_gaussians']):
                fh.write(f"{params[i*3]:.6f}\t{params[i*3+1]:.6f}\t{params[i*3+2]:.6f}\n")

        # ── Diagnostic plot ────────────────────────────────────────────────
        if pdf is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, y, 'k-', lw=0.8, label='Data')
            if result['params']:
                ax.plot(x, multi_gaussian(x, *result['params']),
                        'b-', lw=1.5, label=f'Fit (N={result["n_gaussians"]})')
                ax.axvline(result['envelope_left'],  color='r', ls='--', lw=1.2,
                           label='Envelope')
                ax.axvline(result['envelope_right'], color='r', ls='--', lw=1.2)
            ax.set_title(f"{base}   χ²_red={result['chi2_red']:.2f}", fontsize=9)
            ax.set_xlabel('Sample')
            ax.set_ylabel('S/N')
            ax.legend(fontsize=7)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    if pdf is not None:
        pdf.close()
        print(f"[fit] Diagnostic plots → {plot_pdf}")

    print(f"[fit] Done.  Fitted {len(all_results)} pulses.")
    return all_results


# ── 5. FLUENCE ────────────────────────────────────────────────────────────────

def step_fluence(fit_dir, samp_rate=0.025, sefd=17.0, bw_hz=400e6, n_pol=2,
                 output='pulse_properties.txt'):
    """
    Read every *_fit.txt, load its companion pulse txt, measure fluence within
    the fitted envelope, and write a summary table.

    Output columns: pulse_id  width_ms  fluence_jy_s  n_gaussians  chi2_red
    """
    fit_files = sorted(glob.glob(os.path.join(fit_dir, '*_fit.txt')))
    if not fit_files:
        print(f"[fluence] No *_fit.txt files found in {fit_dir}")
        return

    rows = []
    n_ok = 0

    for fit_file in fit_files:
        base       = os.path.basename(fit_file).replace('_fit.txt', '')
        pulse_file = os.path.join(fit_dir, f"{base}.txt")

        # Parse fit summary header
        env_left = env_right = chi2_red = n_gauss = np.nan
        try:
            with open(fit_file) as fh:
                for line in fh:
                    if 'envelope_left'  in line:
                        env_left  = float(line.split('=')[1].split()[0])
                    elif 'envelope_right' in line:
                        env_right = float(line.split('=')[1].split()[0])
                    elif 'chi2_red'     in line:
                        chi2_red  = float(line.split('=')[1].split()[0])
                    elif 'n_gaussians'  in line:
                        n_gauss   = int(line.split('=')[1].split()[0])
        except Exception as e:
            print(f"  [fluence] Could not parse {fit_file}: {e}")
            rows.append([base, np.nan, np.nan, np.nan, np.nan])
            continue

        # Load pulse data
        try:
            raw = np.loadtxt(pulse_file)
            x   = raw[:, 0]
            y   = raw[:, 1]
        except Exception as e:
            print(f"  [fluence] Could not load {pulse_file}: {e}")
            rows.append([base, np.nan, np.nan, n_gauss, chi2_red])
            continue

        fluence, width_ms = measure_fluence_from_envelope(
            x, y, env_left, env_right,
            samp_rate_ms=samp_rate,
            sefd=sefd, bw_hz=bw_hz, n_pol=n_pol,
        )

        rows.append([base, width_ms, fluence, n_gauss, chi2_red])
        n_ok += 1

    # Write output
    header = "# pulse_id\twidth_ms\tfluence_jy_s\tn_gaussians\tchi2_red"
    with open(output, 'w') as fout:
        fout.write(header + '\n')
        for row in rows:
            pid, wms, fl, ng, chi2 = row
            try:
                fout.write(f"{pid}\t{wms:.4f}\t{fl:.8f}\t{int(ng)}\t{chi2:.4f}\n")
            except (ValueError, TypeError):
                fout.write(f"{pid}\tnan\tnan\tnan\tnan\n")

    print(f"[fluence] Measured {n_ok}/{len(fit_files)} pulses → {output}")
    return rows


# ── 6. ENERGY ─────────────────────────────────────────────────────────────────

def step_energy(properties_file, distance_cm=3e21, sefd=17.0, bw_hz=250e6,
                n_pol=2, output='pulse_energies.txt'):
    """
    Convert fluence (Jy·s) from the properties file into energy (ergs).

    Reads the pulse_properties.txt written by step_fluence.
    """
    data = np.loadtxt(properties_file, dtype=str, comments='#')
    if data.ndim == 1:
        data = data[np.newaxis, :]

    pulse_ids  = data[:, 0]
    fluence    = data[:, 2].astype(float)   # Jy·s

    energy = fluence * bw_hz * 1e-23 * 4.0 * np.pi * distance_cm**2  # ergs

    header = "# pulse_id\tfluence_jy_s\tenergy_ergs"
    with open(output, 'w') as fout:
        fout.write(header + '\n')
        for pid, fl, en in zip(pulse_ids, fluence, energy):
            fout.write(f"{pid}\t{fl:.8f}\t{en:.6e}\n")

    print(f"[energy] Saved energies for {len(energy)} pulses → {output}")
    return energy


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    """Execute read → clean → extract → fit → fluence sequentially."""
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # ── 1. Read
    raw_file = os.path.join(out, 'timeseries_raw.txt')
    step_read(args.fil, args.dm, args.block, output=raw_file)

    # ── 2. Clean
    pairs = []
    if args.bad_channels:
        bc = args.bad_channels
        pairs = [(bc[i], bc[i+1]) for i in range(0, len(bc)-1, 2)]
    cleaned_file = os.path.join(out, 'timeseries_cleaned.txt')
    step_clean(raw_file, bad_channel_pairs=pairs,
               samp_rate=args.samp_rate, output=cleaned_file)

    # ── 3. Extract
    pulse_dir = os.path.join(out, 'pulses')
    step_extract(cleaned_file,
                 distance=args.distance,
                 prominence=args.prominence,
                 half_width=args.half_width,
                 save_dir=pulse_dir,
                 prefix=args.prefix)

    # ── 4. Fit
    pdf_path = os.path.join(out, 'fit_diagnostics.pdf')
    step_fit(pulse_dir,
             samp_rate=args.samp_rate,
             chi2_threshold=args.chi2_threshold,
             max_gaussians=args.max_gaussians,
             sn_threshold=args.sn_threshold,
             output_dir=pulse_dir,
             plot_pdf=pdf_path)

    # ── 5. Fluence
    props_file = os.path.join(out, 'pulse_properties.txt')
    step_fluence(pulse_dir,
                 samp_rate=args.samp_rate,
                 sefd=args.sefd,
                 bw_hz=args.bw,
                 n_pol=args.n_pol,
                 output=props_file)

    print(f"\n[run] Pipeline complete.  Results in: {out}/")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog='stratos.py',
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run  stratos.py <command> --help  for per-command options.",
    )
    sub = parser.add_subparsers(dest='command', metavar='<command>')

    # ── Shared options ────────────────────────────────────────────────────────
    def add_samp_rate(p):
        p.add_argument('--samp-rate', type=float, default=0.025,
                       metavar='MS',
                       help='Time resolution in ms per sample (default: 0.025)')

    def add_radiometer(p):
        p.add_argument('--sefd', type=float, default=17.0,
                       help='System equivalent flux density in Jy (default: 17)')
        p.add_argument('--bw',   type=float, default=400e6,
                       help='Bandwidth in Hz (default: 400e6)')
        p.add_argument('--n-pol', type=int, default=2,
                       help='Number of polarisations (default: 2)')

    # ── read ──────────────────────────────────────────────────────────────────
    p_read = sub.add_parser('read', help='Read & dedisperse a filterbank file',
                             description='Read a .fil file, dedisperse, and save the raw 2-D block.')
    p_read.add_argument('--fil',   required=True, help='Path to .fil file')
    p_read.add_argument('--dm',    type=float, required=True, help='Dispersion measure (pc/cm³)')
    p_read.add_argument('--block', type=int,   required=True, help='Number of samples to read')
    p_read.add_argument('-o', '--output', default='timeseries_raw.txt',
                        help='Output filename (default: timeseries_raw.txt)')

    # ── clean ─────────────────────────────────────────────────────────────────
    p_clean = sub.add_parser('clean', help='RFI-clean channels and produce normalised timeseries',
                              description='Mask bad channel ranges, average in frequency, normalise.')
    p_clean.add_argument('--ts',       required=True, help='Input raw timeseries (2-D txt)')
    p_clean.add_argument('--bad-channels', type=int, nargs='*', metavar='N',
                         help='Pairs of channel indices to mask: --bad-channels 0 50 200 300')
    add_samp_rate(p_clean)
    p_clean.add_argument('-o', '--output', default='timeseries_cleaned.txt',
                         help='Output filename (default: timeseries_cleaned.txt)')

    # ── extract ───────────────────────────────────────────────────────────────
    p_ext = sub.add_parser('extract', help='Extract individual pulses from a timeseries',
                            description='Detect peaks and save pulse windows to individual files.')
    p_ext.add_argument('--ts',          required=True, help='Cleaned timeseries (2-col txt)')
    p_ext.add_argument('--distance',    type=int,   default=160,
                       help='Min sample separation between peaks (default: 160)')
    p_ext.add_argument('--prominence',  type=float, default=8.0,
                       help='Required peak prominence in S/N units (default: 8)')
    p_ext.add_argument('--half-width',  type=int,   default=200,
                       help='Samples to take either side of each peak (default: 200)')
    p_ext.add_argument('--save-dir',    default='./pulses',
                       help='Directory for output pulse files (default: ./pulses)')
    p_ext.add_argument('--prefix',      default='pulse',
                       help='Filename prefix for pulses (default: pulse)')

    # ── fit ───────────────────────────────────────────────────────────────────
    p_fit = sub.add_parser('fit',
                            help='Multi-Gaussian fit all pulse files (chi-square minimisation)',
                            description=(
                                'Iteratively fit Gaussians to each pulse until χ²_red ≤ threshold. '
                                'Saves per-pulse *_fit.txt files and optional diagnostic PDF.'
                            ))
    p_fit.add_argument('--pulse-dir',     required=True,
                       help='Directory containing pulse txt files')
    add_samp_rate(p_fit)
    p_fit.add_argument('--chi2-threshold', type=float, default=1.5,
                       help='Reduced chi-square convergence threshold (default: 1.5)')
    p_fit.add_argument('--max-gaussians',  type=int,   default=10,
                       help='Maximum number of Gaussian components (default: 10)')
    p_fit.add_argument('--sn-threshold',   type=float, default=3.0,
                       help='S/N threshold for peak seeding (default: 3.0)')
    p_fit.add_argument('--output-dir',     default=None,
                       help='Where to save fit files (default: same as --pulse-dir)')
    p_fit.add_argument('--plot-pdf',       default=None,
                       help='If set, write diagnostic plots to this PDF path')

    # ── fluence ───────────────────────────────────────────────────────────────
    p_fl = sub.add_parser('fluence',
                           help='Measure fluence & width for all pulses from fit results',
                           description=(
                               'Reads *_fit.txt files, integrates S/N within the fitted envelope, '
                               'and writes pulse_properties.txt with width (ms) and fluence (Jy·s).'
                           ))
    p_fl.add_argument('--fit-dir', required=True,
                      help='Directory containing *_fit.txt and companion pulse txt files')
    add_samp_rate(p_fl)
    add_radiometer(p_fl)
    p_fl.add_argument('-o', '--output', default='pulse_properties.txt',
                      help='Output summary file (default: pulse_properties.txt)')

    # ── energy ────────────────────────────────────────────────────────────────
    p_en = sub.add_parser('energy',
                           help='Convert fluence → energy (ergs)',
                           description='Reads pulse_properties.txt and writes pulse energies.')
    p_en.add_argument('--properties', required=True,
                      help='Path to pulse_properties.txt from the fluence step')
    p_en.add_argument('--distance',   type=float, default=3e21,
                      help='Distance to source in cm (default: 3e21 ~ 1 kpc)')
    add_radiometer(p_en)
    p_en.add_argument('-o', '--output', default='pulse_energies.txt',
                      help='Output file (default: pulse_energies.txt)')

    # ── run (full pipeline) ───────────────────────────────────────────────────
    p_run = sub.add_parser('run', help='Run the full pipeline end-to-end',
                            description='Execute: read → clean → extract → fit → fluence')
    p_run.add_argument('--fil',   required=True, help='Path to .fil file')
    p_run.add_argument('--dm',    type=float, required=True, help='Dispersion measure (pc/cm³)')
    p_run.add_argument('--block', type=int,   required=True, help='Number of samples to read')
    p_run.add_argument('--bad-channels', type=int, nargs='*', metavar='N',
                       help='Pairs of channel indices to mask')
    add_samp_rate(p_run)
    p_run.add_argument('--distance',    type=int,   default=160,
                       help='Min sample separation between peaks (default: 160)')
    p_run.add_argument('--prominence',  type=float, default=8.0,
                       help='Peak prominence threshold (default: 8)')
    p_run.add_argument('--half-width',  type=int,   default=200,
                       help='Samples each side of peak (default: 200)')
    p_run.add_argument('--prefix',      default='pulse',
                       help='Pulse file prefix (default: pulse)')
    p_run.add_argument('--chi2-threshold', type=float, default=1.5,
                       help='Reduced chi-square threshold (default: 1.5)')
    p_run.add_argument('--max-gaussians',  type=int,   default=10,
                       help='Max Gaussian components per pulse (default: 10)')
    p_run.add_argument('--sn-threshold',   type=float, default=3.0,
                       help='S/N seeding threshold for fitting (default: 3.0)')
    add_radiometer(p_run)
    p_run.add_argument('--output-dir', default='./stratos_output',
                       help='Root output directory (default: ./stratos_output)')

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(BANNER)
    parser = build_parser()
    args   = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == 'read':
        step_read(args.fil, args.dm, args.block, output=args.output)

    elif args.command == 'clean':
        pairs = []
        if args.bad_channels:
            bc = args.bad_channels
            pairs = [(bc[i], bc[i+1]) for i in range(0, len(bc)-1, 2)]
        step_clean(args.ts, bad_channel_pairs=pairs,
                   samp_rate=args.samp_rate, output=args.output)

    elif args.command == 'extract':
        step_extract(args.ts,
                     distance=args.distance,
                     prominence=args.prominence,
                     half_width=args.half_width,
                     save_dir=args.save_dir,
                     prefix=args.prefix)

    elif args.command == 'fit':
        step_fit(args.pulse_dir,
                 samp_rate=args.samp_rate,
                 chi2_threshold=args.chi2_threshold,
                 max_gaussians=args.max_gaussians,
                 sn_threshold=args.sn_threshold,
                 output_dir=args.output_dir,
                 plot_pdf=args.plot_pdf)

    elif args.command == 'fluence':
        step_fluence(args.fit_dir,
                     samp_rate=args.samp_rate,
                     sefd=args.sefd,
                     bw_hz=args.bw,
                     n_pol=args.n_pol,
                     output=args.output)

    elif args.command == 'energy':
        step_energy(args.properties,
                    distance_cm=args.distance,
                    sefd=args.sefd,
                    bw_hz=args.bw,
                    n_pol=args.n_pol,
                    output=args.output)

    elif args.command == 'run':
        run_pipeline(args)


if __name__ == '__main__':
    main()
