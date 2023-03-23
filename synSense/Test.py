import samna;

print("11111111111111")

#  Create a samna node, associate it to the python interpreter, and expose it as a submodule.
#  It also open a remote node named "device_node" as well.
#  The contents of the node are now available as the Python submodule samna.device_node
samna_node = samna.init_samna()

# The endpoints are needed if we need to use visualizer later
sender_endpoint = samna_node.get_sender_endpoint()
receiver_endpoint = samna_node.get_receiver_endpoint()


# Get all connected, but opened devices, supported by Samna.
devices = samna.device.get_unopened_devices();

print("=====:",devices)

# Open the only device
my_board = samna.device.open_device(devices[0]);