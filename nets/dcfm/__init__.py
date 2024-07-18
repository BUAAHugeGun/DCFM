from .basic import PRE, DCFM
from .encoder import PREEncoder, DCFMEncoder
from .decoder import PREDecoder, DCFMDecoder

name2model = {
    "image": PRE,
    "video": DCFM,
    "image_encoder": PREEncoder, 
    "image_decoder": PREDecoder,
    "video_encoder": DCFMEncoder,
    "video_decoder": DCFMDecoder,
}

def get_model(tag:str, **kwargs):
    return name2model[tag](**kwargs)