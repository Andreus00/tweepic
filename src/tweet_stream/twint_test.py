import twint

# Configure
c = twint.Config()
c.Custom["user"] = ["username", "id"]
c.Search = "elonmusk"

# Run
twint.run.Lookup(c)