import sys
sys.path.insert(0, '/home/slr/SLR')

activate_this = '/home/slr/SLR/slrEnv/bin/activate'
with open(activate_this) as file:
    exect(file.read(), dict(__file__ = activate_this))

from app import app as application