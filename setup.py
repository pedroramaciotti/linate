
from distutils.core import setup

setup(
  name = 'linate',         
  version = '0.3',      
  license='MIT',        
  description = 'Language-Independent Network Attitudinal Embedding',   
  author = 'Pedro Ramaciotti Morales',                      
  url = 'https://github.com/pedroramaciotti/linate',   
  keywords = ['graph embedding', 'correnpondence analysis', 'social network analysis'],
  install_requires=[            
          'numpy',
          'pandas',
          'scipy',
      ],
  packages = ["linate"],
  include_package_data=True,
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)