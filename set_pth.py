import os
import site


def set_pth():
    """Creates a .pth file to dynamically add the current project directory to the Python `site-packages` path."""

    # step 1: get the absolute path of the current project
    project_root = os.path.abspath(os.path.dirname(__file__))

    # define the content for the .pth file dynamically
    pth_content = f"import site;site.addsitedir('{project_root}', set());\n"

    # step 2: write the .pth file to the site-packages directory

    # get the site-packages directories
    site_packages = site.getsitepackages()

    # path where the .pth file will go (using the first site-packages directory)
    pth_file_path = os.path.join(site_packages[0], "argbench.pth")

    # write the .pth file to the correct location
    with open(pth_file_path, "w") as f:
        f.write(pth_content)

    print(f"Created {pth_file_path} with project root: {project_root}")


if __name__ == "__main__":
    set_pth()
