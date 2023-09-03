import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Football-Event-Detection-Model"
AUTHOR_USER_NAME = "chau2450"
SRC_REPO = "Football_Event_Detector"
AUTHOR_EMAIL = "chau2450@gmail.com"



setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Event detection model for Football Matches",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    include_package_data=True,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)

