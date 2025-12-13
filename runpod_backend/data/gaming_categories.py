# Gaming and video-specific categories for object detection
# Simplified to visually distinct objects that SAM-3 can actually segment
# NOTE: Common items (person, vehicles, trees, etc.) are in common_categories.py

GAMING_CATEGORIES = [
    # === CHARACTERS (gaming-specific) ===
    {"name": "soldier", "category": "character"},
    {"name": "humanoid", "category": "character"},
    {"name": "zombie", "category": "character"},
    {"name": "robot", "category": "character"},
    {"name": "monster", "category": "character"},

    # === WEAPONS ===
    {"name": "gun", "category": "weapon"},
    {"name": "rifle", "category": "weapon"},
    {"name": "pistol", "category": "weapon"},
    {"name": "shotgun", "category": "weapon"},
    {"name": "sword", "category": "weapon"},
    {"name": "knife", "category": "weapon"},
    {"name": "grenade", "category": "weapon"},

    # === VEHICLES (gaming-specific) ===
    {"name": "tank", "category": "vehicle"},
    {"name": "helicopter", "category": "vehicle"},
    {"name": "jeep", "category": "vehicle"},

    # === ENVIRONMENT (gaming-specific) ===
    {"name": "wall", "category": "environment"},
    {"name": "ladder", "category": "environment"},
    {"name": "crate", "category": "environment"},
    {"name": "barrel", "category": "environment"},

    # === EFFECTS ===
    {"name": "explosion", "category": "effect"},
    {"name": "fire", "category": "effect"},
    {"name": "smoke", "category": "effect"},

    # === ITEMS ===
    {"name": "chest", "category": "item"},
    {"name": "backpack", "category": "item"},
    {"name": "helmet", "category": "item"},
]

# Extract just the names for label matching
GAMING_LABELS = [cat['name'] for cat in GAMING_CATEGORIES]

# Group by category for reference
GAMING_CATEGORIES_BY_TYPE = {}
for cat in GAMING_CATEGORIES:
    category = cat['category']
    if category not in GAMING_CATEGORIES_BY_TYPE:
        GAMING_CATEGORIES_BY_TYPE[category] = []
    GAMING_CATEGORIES_BY_TYPE[category].append(cat['name'])
