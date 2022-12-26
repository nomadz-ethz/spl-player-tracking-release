# Image parameters
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

ORC_SPLITS = {
    "train": [
        "RoboEireann",
        "R-ZWEI KICKERS",
        "NAO Devils",
        "SPQR",
        "NomadZ",
    ],
    "val": [
        "B-Human",
        "Berlin United",
    ],
}

NEW_SPLITS = {
    "all": [
        "2022_07_16_11_30_C_2_1",
        "2022_07_15_16_30_C_1_1",
        "2022_07_16_12_30_C_2_1",
    ]
}

TEAM_NUMBER_TO_COLORS = {
    4: (1,),  # berlin united
    5: (3,),  # bhuman
    12: (2, 3),  # nao devils
    13: (0,),  # htwk
    17: (5,),  # robo eirean
    19: (1,),  # spqr,
    20: (9,),  # hulks
    33: (1,),  # nomadz.
    45: (1,),  # naova
    18: (2,),  # runswift
    24: (9,),  # hulks
}

COLOR_TO_RGB = {
    -1: (128, 128, 128),  # unknown
    0: (224, 160, 0),  # blue
    1: (0, 32, 224),  # red
    2: (0, 224, 224),  # yellow
    3: (0, 0, 0),  # black
    4: (224, 224, 224),  # white
    5: (0, 224, 0),  # green
    6: (0, 128, 224),  # orange
    7: (224, 0, 224),  # purple
    8: (0, 32, 128),  # brown
    9: (64, 64, 64),  # gray
}

CLASSES = [
    "robot",
    "ball",
]

LABEL_MAP = {idx: name for idx, name in enumerate(CLASSES)}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

CSV_ANNOTATION_FIELDS = [
    "filename",
    "label",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "color",
    "number",
]
