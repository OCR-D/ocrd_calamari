import os

class working_directory:
    """Context manager to temporarily change the working directory"""

    def __init__(self, wd):
        self.wd = wd

    def __enter__(self):
        self.old_wd = os.getcwd()
        os.chdir(self.wd)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.old_wd)
