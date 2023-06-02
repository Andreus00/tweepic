# import sys
# sys.path.append('src/tweet_stream/twint')

import twint

# Configure
c = twint.Config()
c.Custom["tweet"] = [1664030208477782018]

# Run
twint.run.Search(c)
