from __future__ import print_function
import os
import platform


class Config(object):
    """docstring for Config"""

    def __init__(self):
        super(Config, self).__init__()

        DATAPATH = os.environ.get('DATAPATH')
        if DATAPATH is None:
            if platform.system() == "Windows" or platform.system() == "Linux":
                # DATAPATH = "D:/data/traffic_flow"
            # elif platform.system() == "Linux":
                DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
            else:
                print("Unsupported/Unknown OS: ", platform.system, "please set DATAPATH")
        self.DATAPATH = DATAPATH
