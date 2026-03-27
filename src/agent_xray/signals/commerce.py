"""Signals for browser commerce and checkout flows."""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from ..schema import AgentStep, AgentTask


class CommerceDetector:
    """Detect shopping, checkout, and payment form activity."""

    name = "commerce"

    PAYMENT_FIELD_KEYWORDS = [
        "card number",
        "credit card",
        "debit card",
        "cvv",
        "cvc",
        "security code",
        "expir",
        "billing address",
        "cardholder",
        "payment method",
    ]
    PAYMENT_FIELDS_STRONG = [
        "card number",
        "cvv",
        "cvc",
        "security code",
        "expir",
        "cardholder",
        "card details",
    ]
    CHECKOUT_KEYWORDS = [
        "checkout",
        "place order",
        "complete order",
        "order summary",
        "proceed to payment",
        "pay now",
    ]
    CART_KEYWORDS = [
        "your cart",
        "shopping cart",
        "cart summary",
        "subtotal",
        "view cart",
        "items in cart",
    ]
    REAL_FILL_PATTERNS = [
        r"card",
        r"cvv",
        r"cvc",
        r"expir",
        r"security",
        r"address",
        r"zip",
        r"postal",
        r"city",
        r"state",
        r"first.?name",
        r"last.?name",
        r"email",
        r"phone",
    ]
    TERMINAL_URL_PATTERNS = [
        r"/payment",
        r"/checkout",
        r"/billing",
        r"/order/confirm",
        r"/order/review",
        r"/place.order",
        r"/confirmation",
    ]

    def detect_step(self, step: AgentStep) -> dict[str, bool]:
        page_url = self._page_url(step)
        result_lower = (step.tool_result or "")[:5000].lower()
        tool_input_text = json.dumps(step.tool_input, sort_keys=True).lower()
        combined_fill_text = (
            f"{step.tool_input.get('text', '')} {step.tool_input.get('fields', '')}".lower()
        )
        is_form_fill = "fill" in step.tool_name.lower() or "fields" in step.tool_input
        is_real_fill = False
        if is_form_fill and any(
            re.search(pattern, tool_input_text) for pattern in self.REAL_FILL_PATTERNS
        ):
            is_real_fill = True
        if re.search(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}", combined_fill_text):
            is_real_fill = True
        if re.search(r"@[\w.-]+\.\w+", combined_fill_text):
            is_real_fill = True
        if re.search(r"\b\d{5}\b", combined_fill_text) and "ref" in step.tool_input:
            is_real_fill = True

        is_payment_fill = is_form_fill and any(
            keyword in f"{tool_input_text} {result_lower}" for keyword in self.PAYMENT_FIELDS_STRONG
        )
        has_payment_fields = any(keyword in result_lower for keyword in self.PAYMENT_FIELD_KEYWORDS)
        has_payment_fields_strong = any(
            keyword in result_lower for keyword in self.PAYMENT_FIELDS_STRONG
        )
        return {
            "is_payment_page": self._is_payment_url(page_url),
            "is_checkout_page": self._is_checkout(page_url, result_lower),
            "is_cart_page": self._is_cart(page_url, result_lower),
            "is_form_fill": is_form_fill,
            "is_real_fill": is_real_fill,
            "is_payment_fill": is_payment_fill,
            "has_payment_fields": has_payment_fields,
            "has_payment_fields_strong": has_payment_fields_strong,
        }

    def summarize(
        self, task: AgentTask, step_signals: list[dict[str, bool]]
    ) -> dict[str, int | bool]:
        form_fill_count = sum(1 for signals in step_signals if signals["is_form_fill"])
        real_fill_count = sum(1 for signals in step_signals if signals["is_real_fill"])
        payment_fill_count = sum(1 for signals in step_signals if signals["is_payment_fill"])
        url_has_terminal = any(signals["is_payment_page"] for signals in step_signals)
        payment_fields_confirmed = any(
            signals["has_payment_fields_strong"] for signals in step_signals
        )
        reached_payment = payment_fields_confirmed or (
            any(signals["has_payment_fields"] for signals in step_signals) and url_has_terminal
        )
        step_count = len(step_signals)
        suspiciously_short = step_count <= 2 and reached_payment
        return {
            "reached_payment": reached_payment,
            "reached_checkout": any(signals["is_checkout_page"] for signals in step_signals),
            "reached_cart": any(signals["is_cart_page"] for signals in step_signals),
            "form_fill_count": form_fill_count,
            "payment_fill_count": payment_fill_count,
            "fill_count": form_fill_count,
            "real_fill_count": real_fill_count,
            "payment_fields_confirmed": payment_fields_confirmed,
            "url_has_terminal": url_has_terminal,
            "suspiciously_short": suspiciously_short,
        }

    def _is_payment_url(self, url: str | None) -> bool:
        path = self._url_path(url)
        return any(re.search(pattern, path) for pattern in self.TERMINAL_URL_PATTERNS)

    def _is_checkout(self, url: str | None, result_lower: str) -> bool:
        path = self._url_path(url)
        return "checkout" in path or any(
            keyword in result_lower for keyword in self.CHECKOUT_KEYWORDS
        )

    def _is_cart(self, url: str | None, result_lower: str) -> bool:
        path = self._url_path(url)
        return "cart" in path or any(keyword in result_lower for keyword in self.CART_KEYWORDS)

    def _url_path(self, url: str | None) -> str:
        if not url:
            return ""
        return urlparse(url).path.lower()

    def _page_url(self, step: AgentStep) -> str | None:
        page_url = getattr(step, "page_url", None)
        if page_url:
            return str(page_url)
        browser = getattr(step, "browser", None)
        browser_page_url = getattr(browser, "page_url", None)
        return str(browser_page_url) if browser_page_url else None
