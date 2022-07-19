## Target -> Art
# Train Source Model
python image_source.py --s 1 2 3 --t 0

# Train Sticker Branch
python image_source_sticker.py --s 1 2 3 --t 0

# Adaptation
python image_target.py --s 1 2 3 --t 0


## Target -> Clipart
# Train Source Model
python image_source.py --s 0 2 3 --t 1

# Train Sticker Branch
python image_source_sticker.py --s 0 2 3 --t 1

# Adaptation
python image_target.py --s 0 2 3 --t 1


## Target -> Product
# Train Source Model
python image_source.py --s 0 1 3 --t 2

# Train Sticker Branch
python image_source_sticker.py --s 0 1 3 --t 2

# Adaptation
python image_target.py --s 0 1 3 --t 2


## Target -> RealWorld
# Train Source Model
python image_source.py --s 0 1 2 --t 3

# Train Sticker Branch
python image_source_sticker.py --s 0 1 2 --t 3

# Adaptation
python image_target.py --s 0 1 2 --t 3