"""Use ordinary Python errors in callable workflows, independent of argparse."""


class ConfigError:
    @staticmethod
    def error(message):
        raise ValueError(message)
