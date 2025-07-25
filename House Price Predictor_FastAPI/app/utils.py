def preprocess(features):
    return [
        features.area,
        features.bedrooms,
        features.bathrooms,
        features.stories,
        features.parking
    ]
