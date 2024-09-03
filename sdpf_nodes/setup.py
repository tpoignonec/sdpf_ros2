# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (tpoignonec@unistra.fr)


from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'sdpf_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ('share/ament_index/resource_index/packages',
        #    ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tpoignonec',
    maintainer_email='tpoignonec@unistra.fr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stiffness_estimation_node = sdpf_nodes.stiffness_estimation_node:main',
            'ppf_node = sdpf_nodes.ppf_node:main',
            'sdpf_node = sdpf_nodes.sdpf_node:main',
            'bednarczyk_node = sdpf_nodes.bednarczyk_node:main'
        ],
    },
)
