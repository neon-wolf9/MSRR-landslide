SEEDS=[7,11,21]
RAIN_FIELDS=['rain_30m_mm','rain_1h_mm','rain_3h_mm','rain_6h_mm','rain_12h_mm','rain_24h_mm','rain_48h_mm','rain_72h_mm','rain_120h_mm','api_k090_step30m_120h']
STATIC_CONTINUOUS=91
STATIC_CATEGORICAL=1
DYNAMIC_SHAPE=(70,10)
REFERENCE_TIME_UTC='2018-07-06T11:00:00Z'
FUTURE_STEPS_FORBIDDEN=122
BOOTSTRAP_REPLICATES=2000
BOOTSTRAP_SEED=20260706
FULL_PARAMETER_COUNT=86311
TRIGGER_COUNT=83
FORBIDDEN_X_TOKENS=['centroid_x','centroid_y','centroid_lon','centroid_lat','label','y_pair','pair_set_id','pair_edge_id','outer_fold','spatial_group','sample_role','control_rank','matching','inventory','gsi_only','future']
INNER_RULE='test=outer; validation=(outer+1)%5; train=remaining_three'
SELECTION_PRIORITY=['pair_set_concordance','edge_wise_ranking_accuracy','median_min_gap','auprc','simpler_configuration']
MAX_CONFIGS={'BASE-01':10,'BASE-02':12,'BASE-03':20,'BASE-04':20,'BASE-05':16,'BASE-06':16}
