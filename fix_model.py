# fix_model.py
# Run once: python fix_model.py

with open('model.py', 'r') as f:
    content = f.read()

# Fix 1: xrange → range (already done but safe to rerun)
content = content.replace('xrange', 'range')

# Fix 2: .items()[i] → list(.items())[i]
content = content.replace(
    "self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]",
    "list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]"
)

with open('model.py', 'w') as f:
    f.write(content)

print("model.py fixed successfully")