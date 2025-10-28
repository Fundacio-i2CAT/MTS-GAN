"""Model Factory"""
import architecture.mts_gan.generator as mb_gen
import architecture.mts_gan.discriminator as mb_disc
import architecture.tts_gan.generator as tts_gen
import architecture.tts_gan.discriminator as tts_disc


class ModelFactory:
    """Model Factory"""
    @staticmethod
    def create_model(config: dict):
        """Create Model"""
        architecture = config["type"]

        if architecture == "tts-gan":
            gen_net = tts_gen.Generator(config["network"]["generator"])
            disc_net = tts_disc.Discriminator(config["network"]["discriminator"])
            return gen_net, disc_net
        elif architecture == "mts-gan":
            gen_net = mb_gen.Generator(config["network"]["generator"])
            disc_net = mb_disc.Discriminator(config["network"]["discriminator"])
            return gen_net, disc_net
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
