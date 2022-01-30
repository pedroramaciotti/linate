from distutils.core import setup

setup(
  name = 'linate',         
  packages = ['Language-Independent Network Attitudinal Embedding'],   
  version = '0.1',      
  license='MIT',        
  description = 'A Python module to embed social networks in attitudinal opinion spaces',   
  author = 'Pedro Ramaciotti Morales',                  
  author_email = 'pedro.ramaciotti@gmail.com',     
  url = 'https://pedroramaciotti.github.io',   
  download_url = 'https://github.com/pedroramaciotti/linate/archive/v_0.1.tar.gz',
  keywords = ['graph embedding', 'correnpondence analysis', 'social network analysis'],
  install_requires=[            
          'numpy',
          'pandas',
          'scipy',
      ],
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