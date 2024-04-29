# -*- coding: utf-8 -*- 
# __author__: Adarsh Kalikadien #
from trading.agent import Agent


def main():
    agent = Agent(1396735200)
    agent.train()
    agent.set_timestamp(1367186400)
    agent.test()


if __name__ == "__main__":
    main()
