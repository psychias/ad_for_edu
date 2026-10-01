"""The strategy registry: closed names, checked parameters, no silent fallbacks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from ad_for_edu.core.errors import (
    DuplicateStrategyError,
    StrategyConfigError,
    UnknownStrategyError,
)
from ad_for_edu.core.registry import Registry, catalogue
from ad_for_edu.core.strategy import StrategySpec


class Greeting(ABC):
    @abstractmethod
    def greet(self, who: str) -> str: ...


def make_registry() -> Registry[Greeting]:
    registry: Registry[Greeting] = Registry("greeting", Greeting, listed=False)

    @registry.register("plain")
    class Plain(Greeting):
        def greet(self, who: str) -> str:
            return f"hello {who}"

    @registry.register("loud")
    class Loud(Greeting):
        def __init__(self, marks: int = 1) -> None:
            self.marks = marks

        def greet(self, who: str) -> str:
            return f"HELLO {who.upper()}" + "!" * self.marks

    return registry


def test_a_registered_strategy_is_built_by_name():
    registry = make_registry()
    assert registry.create("plain").greet("ada") == "hello ada"
    assert registry.create("loud", marks=3).greet("ada") == "HELLO ADA!!!"


def test_the_name_is_recorded_on_the_class():
    registry = make_registry()
    assert registry.get("loud").strategy_name == "loud"


def test_an_unknown_name_fails_and_lists_the_known_ones():
    registry = make_registry()
    with pytest.raises(UnknownStrategyError) as caught:
        registry.create("polite")
    message = str(caught.value)
    assert "polite" in message
    assert "loud" in message and "plain" in message


def test_an_unknown_name_is_a_lookup_error():
    with pytest.raises(LookupError):
        make_registry().get("polite")


def test_a_duplicate_name_is_refused():
    registry = make_registry()
    with pytest.raises(DuplicateStrategyError):

        @registry.register("plain")
        class Again(Greeting):
            def greet(self, who: str) -> str:
                return who


def test_a_class_outside_the_family_is_refused():
    registry = make_registry()
    with pytest.raises(StrategyConfigError):

        @registry.register("stranger")
        class Stranger:  # does not derive from Greeting
            pass


def test_an_unknown_parameter_fails_and_lists_the_accepted_ones():
    registry = make_registry()
    with pytest.raises(StrategyConfigError) as caught:
        registry.create("loud", mark=3)
    message = str(caught.value)
    assert "mark" in message and "marks" in message


def test_a_strategy_without_parameters_refuses_any():
    with pytest.raises(StrategyConfigError) as caught:
        make_registry().create("plain", marks=1)
    assert "no parameters" in str(caught.value)


def test_a_settings_entry_may_be_a_bare_name_or_a_mapping():
    registry = make_registry()
    assert registry.create_from("plain").greet("x") == "hello x"
    built = registry.create_from({"name": "loud", "params": {"marks": 2}})
    assert built.greet("x") == "HELLO X!!"
    assert registry.create_from(StrategySpec("loud")).greet("x") == "HELLO X!"


def test_a_list_of_entries_keeps_its_order():
    registry = make_registry()
    built = registry.create_all(["loud", "plain", {"name": "loud", "params": {"marks": 2}}])
    assert [type(item).__name__ for item in built] == ["Loud", "Plain", "Loud"]


@pytest.mark.parametrize(
    "entry",
    [
        {"params": {"marks": 2}},
        {"name": "loud", "parameters": {"marks": 2}},
        {"name": "loud", "params": [2]},
        "",
        42,
    ],
)
def test_a_malformed_entry_is_refused(entry):
    with pytest.raises(StrategyConfigError):
        StrategySpec.parse(entry)


def test_a_listed_family_appears_in_the_catalogue_once():
    family = "test family for the catalogue"
    registry: Registry[Greeting] = Registry(family, Greeting)
    assert catalogue()[family] is registry
    with pytest.raises(DuplicateStrategyError):
        Registry(family, Greeting)


def test_membership_and_size():
    registry = make_registry()
    assert "plain" in registry and "polite" not in registry
    assert len(registry) == 2
    assert registry.names() == ("loud", "plain")
