"""The client every stage calls models through.

A client exists only after the spend of its stage was approved: its constructor
takes the approval token, and nothing else can stand in for it. Before each call
the client checks that the model may hold the role it is asked to play and can
take what the request carries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..core.errors import ContractError, SpendNotApprovedError
from ..core.spend import SpendApproval
from .base import PROVIDERS, LLMProvider, LLMReply, LLMRequest, TerminalProviderError
from .catalog import ModelCatalog


class LLMClient:
    """Routes requests to the providers of their models."""

    def __init__(
        self,
        approval: SpendApproval,
        catalog: ModelCatalog,
        providers: Mapping[str, LLMProvider] | None = None,
    ) -> None:
        if not isinstance(approval, SpendApproval):
            raise SpendNotApprovedError(
                "a model client needs the approval issued after the spend estimate was shown"
            )
        self.approval = approval
        self.catalog = catalog
        self._providers: dict[str, LLMProvider] = dict(providers or {})
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def provider(self, name: str) -> LLMProvider:
        """The provider called `name`, built on first use."""
        if name not in self._providers:
            strategy, params = self.catalog.provider_spec(name)
            self._providers[name] = PROVIDERS.create(strategy, **params)
        return self._providers[name]

    def _prepare(self, request: LLMRequest, role: str) -> tuple[LLMProvider, str]:
        entry = self.catalog.get(request.model)
        entry.require_role(role)
        entry.require_modalities(request.modalities)
        provider = self.provider(entry.provider)
        if provider.paid and self.approval.total <= 0 and not entry.free:
            raise SpendNotApprovedError(
                f"model {entry.key!r} is paid, but the approval of stage "
                f"{self.approval.stage!r} covers no spend"
            )
        return provider, entry.model_id

    def generate(self, request: LLMRequest, *, role: str) -> LLMReply:
        """Make one call with the model acting as `role`."""
        provider, model_id = self._prepare(request, role)
        reply = provider.generate(request, model_id)
        self._count(reply)
        return reply

    def generate_many(
        self, requests: Sequence[LLMRequest], *, role: str
    ) -> list[LLMReply | Exception]:
        """Make several calls to one model. Replies come back in the order of the requests."""
        if not requests:
            return []
        models = {request.model for request in requests}
        if len(models) != 1:
            raise ContractError(
                f"a batch goes to one model; these requests name {sorted(models)}"
            )
        provider, model_id = self._prepare(requests[0], role)
        for request in requests[1:]:
            self.catalog.get(request.model).require_modalities(request.modalities)
        replies = provider.generate_many(list(requests), model_id)
        if len(replies) != len(requests):
            raise ContractError(
                f"provider returned {len(replies)} replies for {len(requests)} requests"
            )
        for reply in replies:
            if isinstance(reply, TerminalProviderError):
                raise reply
            if isinstance(reply, LLMReply):
                self._count(reply)
        return replies

    def _count(self, reply: LLMReply) -> None:
        self.calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens

    def spent(self, model: str) -> float:
        """What the counted tokens cost at the listed price of `model`.

        This is an estimate from reported usage. A service may read an input it does
        not report as tokens, a clip for example, so the bill can be higher.
        """
        return self.catalog.get(model).cost(self.prompt_tokens, self.completion_tokens)
