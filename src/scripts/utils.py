import argparse
import configparser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=str,
                        help='configuration file', default='configuration.ini')
    args = parser.parse_args()
    return args


def get_config(configfile):
    """
    read config file

    Parameters:
    ----------
    configfile:str, configuration file

    Returns
    -------
    config : ConfigParser object
        configuration file.

    """
    config = configparser.ConfigParser()
    config.read(configfile)
    return config
