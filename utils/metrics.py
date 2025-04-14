import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from tsfresh.feature_extraction import extract_features, MinimalFCParameters

import utils.utils as ut


def compute_fid(real_features, fake_features):
    """Compute the Fréchet Inception Distance (FID) score."""
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    cov_sqrt, _ = sqrtm(sigma_real @ sigma_fake, disp=False)
    
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    
    fid = np.sum((mu_real - mu_fake)**2) + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return fid


def average_cosine_similarity(real_signals, synthetic_signals):
    """Compute the average Cosine Similarity between the synthetic signals and the real ones"""
    similarities = cosine_similarity(synthetic_signals, real_signals)
    mean_similarities = np.mean(similarities, axis=1)
    return np.mean(mean_similarities)


def average_jensen_shannon_distance(real_signals, synthetic_signals):
    distances = []
    for syn_seq in synthetic_signals:
        syn_seq = normalize_for_jsd(syn_seq)
        js_per_real = []
        for real_seq in real_signals:
            real_seq = normalize_for_jsd(real_seq)
            js_div = jensenshannon(real_seq, syn_seq)
            js_per_real.append(js_div)

        mean_jsd = np.mean(js_per_real)
        distances.append(mean_jsd)
    return np.mean(distances)


def normalize_for_jsd(seq):
    """For jensen shannon values must be non negative and normalized."""
    seq = seq - np.min(seq) if np.min(seq) < 0 else seq   
    seq = seq + 1e-8   
    return seq / np.sum(seq)   


def average_fid_score(real_signals, synthetic_signals):
 
    tsfresh_settings = MinimalFCParameters()

    real_feature_df = ut.convert_to_tsfresh_format(real_signals)
    synthetic_feature_df = ut.convert_to_tsfresh_format(synthetic_signals)

    real_features = extract_features(
        real_feature_df, column_id="id", column_sort="time",
        default_fc_parameters=tsfresh_settings, disable_progressbar=True
    )
    synthetic_features = extract_features(
        synthetic_feature_df, column_id="id", column_sort="time",
        default_fc_parameters=tsfresh_settings, disable_progressbar=True
    )

    fid_score = compute_fid(real_features.values, synthetic_features.values)
     
    return np.mean(fid_score)


def compute_metrics(real_data, synthetic_data):
    # shape (num_seq, seq_len, features)
    real_data, synthetic_data = np.array(real_data), np.array(synthetic_data)

    metrics = {}
    avg_cosine_list = []
    avg_js_list = []
    avg_fid_list = []

    # feature level
    for i in range(real_data.shape[2]):
        cosine_sim = average_cosine_similarity(real_data[:, :, i], synthetic_data[:, :, i])
        js_div = average_jensen_shannon_distance(real_data[:, :, i], synthetic_data[:, :, i])
        fid = average_fid_score(real_data[:, :, i], synthetic_data[:, :, i])
        
        metrics[f"feature_{i + 1}"] = {
            "average_cosine_similarity": cosine_sim,
            "jensen_shannon_divergence": js_div,
            "fid_score": fid
        }

        avg_cosine_list.append(cosine_sim)
        avg_js_list.append(js_div)
        avg_fid_list.append(fid)

    # Flatten dictionary
    m = {f"{feature}_{metric}": value for feature, values in metrics.items() for metric, value in values.items()}

    # Add overall/global averages
    # Compares every synthetic signal with 
    m.update({
        "average_cosine_similarity": np.mean(avg_cosine_list),
        "average_jensen_shannon_divergence": np.mean(avg_js_list),
        "average_fid_score": np.mean(avg_fid_list)
    })

    return m
