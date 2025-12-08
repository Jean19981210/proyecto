from scipy.signal import cheby2, filtfilt, butter, savgol_filter, find_peaks

# -------------------------------------------------------------------
# 1) Filter BVP: 4th order Chebyshev Type II bandpass 0.5–10 Hz
# -------------------------------------------------------------------
def preprocess_bvp(bvp, fs):
    nyq = fs / 2
    low, high = 0.5 / nyq, 10 / nyq
    b, a = cheby2(N=4, rs=20, Wn=[low, high], btype='bandpass')
    filtered = filtfilt(b, a, bvp)

    # Detect peaks → Equivalent to pulse interval extraction
    peaks, _ = find_peaks(filtered, distance=0.3 * fs)  # Empirically similar to PPG peak distance
    return filtered, peaks


# -------------------------------------------------------------------
# 2) Filter EDA: Butterworth LPF 1.99 Hz + Savitzky-Golay smoothing
#    Then tonic/phasic separation (approximation)
# -------------------------------------------------------------------
def preprocess_eda(eda, fs):
    nyq = fs / 2
    cutoff = 1.99 / nyq
    b, a = butter(5, cutoff, btype="low")
    eda_low = filtfilt(b, a, eda)

    # Decompose tonic & phasic (approximation used commonly)
    tonic = savgol_filter(eda_low, 101 if len(eda_low) > 101 else len(eda_low)-1, polyorder=3)
    phasic = eda_low - tonic

    return eda_low, tonic, phasic


# -------------------------------------------------------------------
# 3) HR and ACC: no filtering (paper explicitly says so)
# -------------------------------------------------------------------
def preprocess_hr(hr):
    return hr

def preprocess_acc(x, y, z):
    return x, y, z


# -------------------------------------------------------------------
# 4) Apply preprocessing to all channels for 1 subject
# -------------------------------------------------------------------
def apply_full_preprocessing(signals, fs_dict):
    out = {}

    # === EDA ===
    eda = signals["EDA"]
    fs_eda = fs_dict["EDA"]
    eda_low, tonic, phasic = preprocess_eda(eda, fs_eda)
    out["EDA"] = eda_low
    out["EDA_tonic"] = tonic
    out["EDA_phasic"] = phasic
    fs_dict["EDA_tonic"] = fs_eda
    fs_dict["EDA_phasic"] = fs_eda

    # === BVP ===
    bvp = signals["BVP"]
    fs_bvp = fs_dict["BVP"]
    bvp_filt, peaks = preprocess_bvp(bvp, fs_bvp)
    out["BVP"] = bvp_filt
    # out["BVP_peaks"] = peaks  # not time-series; skip for now

    # === HR ===
    out["HR"] = preprocess_hr(signals["HR"])

    # === TEMP === (paper did NOT use TEMP; but keep for model input)
    # out["TEMP"] = signals["TEMP"]

    # === ACC ===
    ax = signals["ACC_x"]; ay = signals["ACC_y"]; az = signals["ACC_z"]
    out["ACC_x"], out["ACC_y"], out["ACC_z"] = preprocess_acc(ax, ay, az)

    return out, fs_dict
