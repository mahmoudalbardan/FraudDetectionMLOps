import argparse
import configparser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=str,
                        help='configuration file', default='configuration.ini')
    parser.add_argument("--retrain", type=str,
                        help='true or false: true corresponds to a retraining '
                             'of the model after performance degradation and false corresponds to the first train', default='false')

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
