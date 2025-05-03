# Generates indexes of all texts in a project.

# This should work by:
# 1. Read ALL pages in the project
# 2. Create list of terms used 
#   2b.? Remove all boring/stop words using AI or otherwise
# 3. Keep track of which pages each term appears in
# 4. Write out a JSON file with the index

import json
from pathlib import Path

