"""
Wrapper for an environment
"""
from authenticators.abstract_authenticator import AbstractAuthenticator


class Agent(AbstractAuthenticator):
    def __init__(self, name):
        super().__init__(name=name)

def step(customer):
    