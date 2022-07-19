## Source -> Art
# Train Source Model
python image_source.py --s 0 --t 1 2 3

# Train Sticker Branch
# A2C
python image_source_sticker.py --s 0 --t 1
# A2P
python image_source_sticker.py --s 0 --t 2
# A2R
python image_source_sticker.py --s 0 --t 3

# Adaptation
# A2C
python image_target.py --s 0 --t 1
# A2P
python image_target.py --s 0 --t 2
# A2R
python image_target.py --s 0 --t 3


## Source -> Clipart
# Train Source Model
python image_source.py --s 1 --t 0 2 3

# Train Sticker Branch
# C2A
python image_source_sticker.py --s 1 --t 0
# C2P
python image_source_sticker.py --s 1 --t 2
# C2R
python image_source_sticker.py --s 1 --t 3

# Adaptation
# C2A
python image_target.py --s 1 --t 0
# C2P
python image_target.py --s 1 --t 2
# C2R
python image_target.py --s 1 --t 3


## Source -> Product
# Train Source Model
python image_source.py --s 2 --t 0 1 3

# Train Sticker Branch
# P2A
python image_source_sticker.py --s 2 --t 0
# P2C
python image_source_sticker.py --s 2 --t 1
# P2R
python image_source_sticker.py --s 2 --t 3

# Adaptation
# P2A
python image_target.py --s 2 --t 0
# P2C
python image_target.py --s 2 --t 1
# P2R
python image_target.py --s 2 --t 3


## Source -> RealWorld
# Train Source Model
python image_source.py --s 3 --t 0 1 2

# Train Sticker Branch
# R2A
python image_source_sticker.py --s 3 --t 0
# R2C
python image_source_sticker.py --s 3 --t 1
# R2P
python image_source_sticker.py --s 3 --t 2

# Adaptation
# R2A
python image_target.py --s 3 --t 0
# R2C
python image_target.py --s 3 --t 1
# R2P
python image_target.py --s 3 --t 2