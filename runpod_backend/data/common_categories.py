# Common object categories for general-purpose object detection
# These are high-frequency, broad categories that work well with SAM-3

COMMON_CATEGORIES = [
    # People
    {"name": "person", "category": "people"},
    {"name": "face", "category": "people"},
    {"name": "hand", "category": "people"},
    {"name": "crowd", "category": "people"},

    # Animals
    {"name": "dog", "category": "animal"},
    {"name": "cat", "category": "animal"},
    {"name": "bird", "category": "animal"},
    {"name": "horse", "category": "animal"},
    {"name": "fish", "category": "animal"},

    # Vehicles
    {"name": "car", "category": "vehicle"},
    {"name": "truck", "category": "vehicle"},
    {"name": "bicycle", "category": "vehicle"},
    {"name": "motorcycle", "category": "vehicle"},
    {"name": "bus", "category": "vehicle"},
    {"name": "train", "category": "vehicle"},
    {"name": "airplane", "category": "vehicle"},
    {"name": "boat", "category": "vehicle"},

    # Architecture & Structures
    {"name": "building", "category": "architecture"},
    {"name": "house", "category": "architecture"},
    {"name": "tower", "category": "architecture"},
    {"name": "bridge", "category": "architecture"},
    {"name": "column", "category": "architecture"},
    {"name": "arch", "category": "architecture"},
    {"name": "door", "category": "architecture"},
    {"name": "window", "category": "architecture"},
    {"name": "roof", "category": "architecture"},
    {"name": "stairs", "category": "architecture"},
    {"name": "fence", "category": "architecture"},
    {"name": "ruins", "category": "architecture"},
    {"name": "monument", "category": "architecture"},
    {"name": "statue", "category": "architecture"},

    # Nature & Landscape
    {"name": "tree", "category": "nature"},
    {"name": "plant", "category": "nature"},
    {"name": "flower", "category": "nature"},
    {"name": "grass", "category": "nature"},
    {"name": "rock", "category": "nature"},
    {"name": "mountain", "category": "nature"},
    {"name": "hill", "category": "nature"},
    {"name": "water", "category": "nature"},
    {"name": "river", "category": "nature"},
    {"name": "lake", "category": "nature"},
    {"name": "ocean", "category": "nature"},
    {"name": "sky", "category": "nature"},
    {"name": "cloud", "category": "nature"},
    {"name": "sand", "category": "nature"},
    {"name": "forest", "category": "nature"},

    # Furniture
    {"name": "chair", "category": "furniture"},
    {"name": "table", "category": "furniture"},
    {"name": "couch", "category": "furniture"},
    {"name": "bed", "category": "furniture"},
    {"name": "desk", "category": "furniture"},
    {"name": "shelf", "category": "furniture"},

    # Electronics
    {"name": "laptop", "category": "electronics"},
    {"name": "phone", "category": "electronics"},
    {"name": "television", "category": "electronics"},
    {"name": "computer", "category": "electronics"},
    {"name": "screen", "category": "electronics"},
    {"name": "camera", "category": "electronics"},

    # Food & Kitchen
    {"name": "food", "category": "food"},
    {"name": "fruit", "category": "food"},
    {"name": "vegetable", "category": "food"},
    {"name": "plate", "category": "food"},
    {"name": "bowl", "category": "food"},

    # Common objects
    {"name": "bottle", "category": "object"},
    {"name": "cup", "category": "object"},
    {"name": "bag", "category": "object"},
    {"name": "book", "category": "object"},
    {"name": "sign", "category": "object"},
    {"name": "light", "category": "object"},
    {"name": "lamp", "category": "object"},
    {"name": "clock", "category": "object"},
    {"name": "mirror", "category": "object"},

    # Sports & Recreation
    {"name": "ball", "category": "sports"},
    {"name": "racket", "category": "sports"},
    {"name": "goal", "category": "sports"},

    # Clothing & Accessories
    {"name": "hat", "category": "clothing"},
    {"name": "shoe", "category": "clothing"},
    {"name": "glasses", "category": "clothing"},

    # Ground & Surfaces
    {"name": "road", "category": "surface"},
    {"name": "path", "category": "surface"},
    {"name": "floor", "category": "surface"},
    {"name": "pavement", "category": "surface"},
]

# Extract just the names for label matching
COMMON_LABELS = [cat['name'] for cat in COMMON_CATEGORIES]
