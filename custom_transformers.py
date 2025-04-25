# custom_transformers.py

class SpatialFeatures:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Only use columns that exist in your data
        features = []
        if 'trip_distance' in X.columns: features.append('trip_distance')
        if 'hour_of_day' in X.columns: features.append('hour_of_day') 
        if 'pickup_zone' in X.columns: features.append('pickup_zone')
        return X[features]  # Use only available features
