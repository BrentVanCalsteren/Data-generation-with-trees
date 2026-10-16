This project started as part of a master thesis, sadly i couldn't complete it in time but is stil getting developed in my free time.

Short summenary of what this projects want to achieve.

This is a project that explores ways to generete synthetic data with decision trees.
It will try to create samples based only on the structure of the tree itself.
It will do this unsupervised.
(So what the splits inside the tree tell use about the data)
The project is currently compleetly writin in python, but i'm debaiting to migrate the more calc expensive functionality to c++
The decision tree library used for this project is called Pydl8.5, which is a python wrapper for DL8.5.

Why DL8.5 and not a other tree library?

In order to generate good new data with trees based on it's structure alone you will need a tree with usefull internal splits.
So for better results you will need the best trees you can fit on given data.
Sadly without restricting the depth you have a endless number of trees and it will be very hard of not impossible to find the best trees.
Most tree algoritms don't care about the best splits in trees and do local greedy splits while keep extending the trees in depth untill criteria are met.
This works but most of the time will result in a lot more complex trees than the tree needs to be. (it could be that there is a tree with the same expresivess but is way less complex in nature)
The Dl8.5 algoritm can garantuee to give you the best possible splits for a tree based on a given depth if given enough time.
(Note: you can set a time limit for faster results, which won't garantuee to return the best tree but still the best solution it found in this time)

Overview of the project files in following folders:

- compare:
TODO: to add

- data
TODO: to add

- samplers
TODO: to add

- trees:
TODO: to add


