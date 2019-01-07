import os
import sys


class Silencer:
    @staticmethod
    def mute():
        sys.stdout = open(os.devnull, 'w')

    @staticmethod
    def unmute():
        sys.stdout = sys.__stdout__
