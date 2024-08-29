from setuptools import setup, find_packages

setup(
    name="Gamma Exposure Analysis",
    version="1.0.0",
    description="A Streamlit app for analyzing gamma exposure.",
    author="Jerrud",
    credit="Sergei Perfiliev",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.27.1",
        "pandas>=2.2.2",
        "numpy>=1.26.4",
        "scipy>=1.14.0",
        "matplotlib>=3.9.2"
    ],
    entry_points={
        'console_scripts': [
            'gamma_exposure_analysis=gamma_app:main',
        ],
    },
)
