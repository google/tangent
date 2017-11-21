# Contributing guidelines

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement (CLA).

  * If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](http://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](http://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

### Contributing code

#### Adding new derivatives

We still have a lot of derivatives we need to write! To add a new derivative for a primitive operation,

- [Read the docs on how to write derivatives in Tangent, and look at some examples](https://github.com/google/tangent/blob/7bf4eaffd646a5906aa15a852f117833d37fb09a/tangent/grads.py#L14-L33)!
- Add your primitive op's reverse-mode derivative to grads.py (example reverse-mode derivative [for np.sin](https://github.com/google/tangent/blob/7bf4eaffd646a5906aa15a852f117833d37fb09a/tangent/grads.py#L230-L232), and for [tf.sin](https://github.com/google/tangent/blob/7bf4eaffd646a5906aa15a852f117833d37fb09a/tangent/tf_extensions.py#L183-L185)).
- Add its forward-mode derivative to tangents.py (example forward-mode derivative for [np.sin](https://github.com/google/tangent/blob/7bf4eaffd646a5906aa15a852f117833d37fb09a/tangent/tangents.py#L144-L146), and for [tf.sin](https://github.com/google/tangent/blob/7bf4eaffd646a5906aa15a852f117833d37fb09a/tangent/tf_extensions.py#L344-L346))
- Add a function using the primitive operation in functions.py. Our tests will pick up on it automatically ([example test function](https://github.com/google/tangent/blob/7bf4eaffd646a5906aa15a852f117833d37fb09a/tests/functions.py#L406-L407)).
- Make sure the tests pass. The tests will be run automatically with Travis once you submit a PR, but it's good to do this locally, so you don't have to wait as long.
```
# Make sure you have pytest installed
pip install pytest
# Run this command from the root of the Tangent project
py.test --short tests
```

#### Adding other new functionality

Tangent is a work-in-progress, so there's a lot of upgrades and tweaks that would be useful. If you've already fixed a bug, or added an enhancement, open a PR, and the team will take a look at it, and help you polish it, conform to style guidelines etc., before we merge it. If you're thinking about embarking on a feature enhancement, open a GitHub issue to start a discussion, or [talk to us on gitter](https://gitter.im/google/tangent).
