import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from sklearn.metrics import mean_squared_error
from tsfeatures import tsfeatures
import pandas as pd
import utils.utils as ut


def compute_mse(real_data, recovered_data) -> dict:
    metrics = {}

    n_samples = real_data.shape[0]
    mse_per_series = []

    for i in range(n_samples):
        real_ts = real_data[i]           
        rec_ts = recovered_data[i]        

        mse = mean_squared_error(real_ts, rec_ts)   
        mse_per_series.append(mse)

    avg_mse = np.mean(mse_per_series)

    metrics['avg_mse'] = avg_mse

    return metrics


def average_fid_score(real_features, fake_features):
    """Compute the Fréchet Inception Distance (FID) score."""
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    cov_sqrt, _ = sqrtm(sigma_real @ sigma_fake, disp=False)

    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = np.sum((mu_real - mu_fake)**2) + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return np.mean(fid)


def average_cosine_similarity(real_signals, synthetic_signals):

    similarities = cosine_similarity(synthetic_signals, real_signals)
    mean_similarities = np.mean(similarities, axis=1)
    return np.mean(mean_similarities)


def average_jensen_shannon_distance(real_signals, synthetic_signals):
    real_arr = np.array(real_signals)
    synthetic_arr = np.array(synthetic_signals)

    # Normalize both arrays beforehand
    real_arr = np.array([normalize_for_jsd(r) for r in real_arr])
    synthetic_arr = np.array([normalize_for_jsd(s) for s in synthetic_arr])

    # Compute pairwise distances
    jsd_matrix = cdist(synthetic_arr, real_arr, metric='jensenshannon')
    distances = np.mean(jsd_matrix, axis=1)

    return np.mean(distances)


def normalize_for_jsd(seq):
    """For jensen shannon values must be non negative and normalized."""
    seq = seq - np.min(seq) if np.min(seq) < 0 else seq
    seq = seq + 1e-8
    return seq / np.sum(seq)


def compute_statistic_metrics(real_signals, synthetic_signals):
    # Metrics to computed with statistics extracted from each signal
    tsfresh_settings = {"median": None,
                        "mean": None,
                        "standard_deviation": None,
                        "variance": None,
                        "root_mean_square": None,
                        "maximum": None,
                        "minimum": None}

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

    metrics = {"fid_statistic": average_fid_score(real_features.values, synthetic_features.values),
               "jensen_shannon_statistic": average_jensen_shannon_distance(real_features.values, synthetic_features.values),
               "cosine_similarity_statistic": average_cosine_similarity(real_features.values, synthetic_features.values)}

    return metrics


def compute_features_metrics(real_signals, synthetic_signals):


    real_feature_df = ut.convert_to_tsfresh_format(real_signals)
    real_feature_df.rename(columns={'id': 'unique_id', 'time':'ds', 'value':'y'}, inplace=True)
    real_feature_df['ds'] = np.tile(pd.date_range(start='2023-01-01', periods=real_signals.shape[1], freq='D'), real_signals.shape[0])

    synthetic_feature_df = ut.convert_to_tsfresh_format(synthetic_signals)
    synthetic_feature_df.rename(columns={'id': 'unique_id', 'time':'ds', 'value':'y'}, inplace=True)
    synthetic_feature_df['ds'] = np.tile(pd.date_range(start='2023-01-01', periods=synthetic_signals.shape[1], freq='D'), real_signals.shape[0])
    
    real_feat = tsfeatures(real_feature_df)
    real_feat = real_feat.fillna(0)
    real_feat = real_feat.drop(columns=["unique_id"])

    syn_feat = tsfeatures(synthetic_feature_df)
    syn_feat = syn_feat.fillna(0)
    syn_feat = syn_feat.drop(columns=["unique_id"])

    metrics = {"fid_temporal": average_fid_score(real_feat, syn_feat),
               "jensen_shannon_temporal": average_jensen_shannon_distance(real_feat, syn_feat),
               "cosine_similarity_temporal": average_cosine_similarity(real_feat, syn_feat)}

    return metrics

def compute_metrics(real_data, synthetic_data):
    # shape (num_seq, seq_len, features)
    real_data, synthetic_data = np.array(real_data), np.array(synthetic_data)

    metrics = {}
    avg_cosine_list = []
    avg_js_list = []
    avg_fid_statistic_list = []
    avg_cosine_statistic_list = []
    avg_js_statistic_list = []
    avg_fid_t_statistic_list = []
    avg_cosine_t_statistic_list = []
    avg_js_t_statistic_list = []

    synthetic_data = np.nan_to_num(synthetic_data, nan=0)
    # feature level
    for i in range(real_data.shape[2]):
        cosine_sim = average_cosine_similarity(real_data[:, :, i], synthetic_data[:, :, i])
        js_div = average_jensen_shannon_distance(real_data[:, :, i], synthetic_data[:, :, i])
        statistic_metrics = compute_statistic_metrics(real_data[:, :, i], synthetic_data[:, :, i])
        temporal_metrics = compute_features_metrics(real_data[:, :, i], synthetic_data[:, :, i])

        metrics[f"feature_{i + 1}"] = {
            "cosine_similarity": cosine_sim,
            "jensen_shannon_divergence": js_div,
            "fid_score_statistic": statistic_metrics["fid_statistic"],
            "cosine_similarity_statistic": statistic_metrics["cosine_similarity_statistic"],
            "jenssen_shannon_statistic_statistic": statistic_metrics["jensen_shannon_statistic"],
            "fid_temporal": temporal_metrics["fid_temporal"],
            "cosine_similarity_temporal": temporal_metrics["cosine_similarity_temporal"],
            "jensen_shannon_temporal": temporal_metrics["jensen_shannon_temporal"]
        }

        avg_cosine_list.append(cosine_sim)
        avg_js_list.append(js_div)
        avg_fid_statistic_list.append(statistic_metrics["fid_statistic"])
        avg_cosine_statistic_list.append(statistic_metrics["cosine_similarity_statistic"])
        avg_js_statistic_list.append(statistic_metrics["jensen_shannon_statistic"])
        avg_fid_t_statistic_list.append(temporal_metrics["fid_temporal"])
        avg_cosine_t_statistic_list.append(temporal_metrics["cosine_similarity_temporal"])
        avg_js_t_statistic_list.append(temporal_metrics["jensen_shannon_temporal"])

    # Flatten dictionary
    m = {f"{feature}_{metric}": value for feature, values in metrics.items() for metric, value in values.items()}

    # Add overall/global averages
    m.update({
        "average_cosine_similarity": np.mean(avg_cosine_list),
        "average_jensen_shannon_divergence": np.mean(avg_js_list),
        "average_fid_score_statistic": np.mean(avg_fid_statistic_list),
        "average_cosine_similarity_statistic": np.mean(avg_cosine_statistic_list),
        "average_jensen_shannon_divergence_statistic": np.mean(avg_js_statistic_list),
        "average_fid_score_temporal": np.mean(avg_fid_t_statistic_list),
        "average_cosine_similarity_temporal": np.mean(avg_cosine_t_statistic_list),
        "average_jensen_shannon_divergence_temporal": np.mean(avg_js_t_statistic_list)

    })

    return m
