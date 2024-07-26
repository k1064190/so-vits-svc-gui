

class PostProcessingManager():
    def __init__(self):
        self.post_processing_modes = ["none", "NSF-HifiGAN", "shallow_diffusion"]
        self.post_processing = None
        self.post_processing_object = None

    def initialize(self, post_processing):
        # TODO
        pass

    def process(self, data):
        return data
