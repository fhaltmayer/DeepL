This is a pytorch implementation of Progressive Growth of Gans: https://arxiv.org/abs/1710.10196. I tried my best at recreating the paper with what was givin within it and a few other references (labeled in code). The calculations seem to be right as the results are working, but there might be some errors due to confusion of the math formulas in the wgan paper. The network structures are not the cleanest, but they are easier to read and debug. Currently running the file is a bit of a mess as I realized there needs to be some changes after getting a bit into training already, changing any of these things woud result in me having to rerun the training from the start which would go over my budget using AWS. Most of the code is labeled so adapting it to your own use should not be too bad.
Things to do:
- [x] Get Working
- [x] Make runable file
- [x] Work with Amazon spot instances
- [ ] Apply mixed precision (currently stuck with using mixed precision with autograd call in gradient penalty)
- [ ] Change network code to use loops

![Training](output.gif)
