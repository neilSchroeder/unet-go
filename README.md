# UNet-go

My first attempt at go: building a UNet from "scratch". 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Instructions on how to install or set up the project.

## Usage

Instructions on how to use or run the project.

## Contributing

Thank you for your interest in contributing to the UNet-go project! We welcome contributions from the community to help improve the project.

To contribute, please follow these guidelines:

1. Fork the repository and create a new branch for your contribution.
2. Make your changes and ensure that they adhere to the project's coding style and guidelines.
3. Write clear and concise commit messages.
4. Test your changes thoroughly to ensure they do not introduce any regressions.
5. Submit a pull request with your changes, providing a detailed description of the changes made and the rationale behind them.

By contributing to the UNet-go project, you agree to license your contributions under the project's [license](#license).

If you have any questions or need further assistance, please feel free to reach out to the project maintainers.

Happy contributing!


## License

The UNet-go project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

You are free to use, modify, and distribute this project under the terms of the MIT License. This license allows you to use the project for personal or commercial purposes, as long as you include the original copyright notice and disclaimer.

For more information about the MIT License, please refer to the [LICENSE](LICENSE) file in the root of the project.


## Notes

The only way for this whole skip connections business to make sense is to apply every filter in every layer to every output feature map in every output. It's a mess and whoever managed to make this happen quickly is a genius, but we can do it.