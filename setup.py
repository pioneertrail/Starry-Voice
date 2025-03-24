from setuptools import setup, find_packages

setup(
    name="starryvoice",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'openai>=1.12.0,<2.0.0',
        'python-dotenv>=1.0.0,<2.0.0',
        'sounddevice>=0.4.6,<0.5.0',
        'soundfile>=0.12.1,<0.13.0',
        'numpy>=1.24.0,<2.0.0',
        'pygame>=2.5.0,<3.0.0',
        'vosk>=0.3.45,<0.4.0',
        'pyttsx3>=2.90,<3.0.0',
    ],
    python_requires='>=3.8',
) 