import sys
sys.path.insert(0, '/home/slr/SLR')

#activate_this = '/root/.local/share/virtualenvs/SLR-okXVGtnt/bin/activate_this.py'
activate_this = '/home/slr/.venvs/SLR-okXVGtnt/bin/activate_this.py'
with open(activate_this) as file:
    exec(file.read(), dict(__file__ = activate_this))

from app import app as application
