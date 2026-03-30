"""Signals for browser commerce and checkout flows."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from ..schema import AgentStep, AgentTask
from ..text_utils import tool_result_text


class CommerceDetector:
    """Detect shopping, checkout, and payment form activity."""

    name = "commerce"

    CONFIDENCE_RANK = {
        "none": 0,
        "keyword_match": 1,
        "url_match": 2,
        "action_sequence": 3,
    }
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
    ADD_TO_CART_ACTION_KEYWORDS = [
        "add to cart",
        "added to cart",
        "add-to-cart",
        "add_to_cart",
        "add to basket",
        "added to basket",
        "add-to-basket",
        "add to bag",
        "added to bag",
    ]
    CHECKOUT_ACTION_KEYWORDS = [
        "proceed to checkout",
        "continue to checkout",
        "go to checkout",
        "checkout",
        "check out",
        "review order",
        "order summary",
        "place order",
        "complete order",
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
    CART_URL_PATTERNS = [
        r"/cart(?:/|$)",
        r"/basket(?:/|$)",
        r"/bag(?:/|$)",
    ]
    CHECKOUT_URL_PATTERNS = [
        r"/check-?out(?:/|$)",
        r"/order(?:/|$)",
        r"/review(?:/|$)",
        r"/billing(?:/|$)",
        r"/shipping(?:/|$)",
        r"/delivery(?:/|$)",
    ]
    PAYMENT_URL_PATTERNS = [
        r"/payment(?:/|$)",
        r"/pay(?:/|$)",
        r"/card(?:/|$)",
        r"/wallet(?:/|$)",
        r"/billing/payment(?:/|$)",
    ]

    def detect_step(self, step: AgentStep) -> dict[str, bool]:
        page_url = self._page_url(step)
        result_lower = tool_result_text(step.tool_result)[:5000].lower()
        tool_input_text = json.dumps(step.tool_input, sort_keys=True).lower()
        combined_text = f"{tool_input_text} {result_lower}"
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

        has_payment_fields = any(keyword in combined_text for keyword in self.PAYMENT_FIELD_KEYWORDS)
        has_payment_fields_strong = any(
            keyword in combined_text for keyword in self.PAYMENT_FIELDS_STRONG
        )
        is_payment_fill = is_form_fill and has_payment_fields_strong
        return {
            "is_payment_page": self._is_payment_url(page_url),
            "is_checkout_page": self._is_checkout_url(page_url),
            "is_cart_page": self._is_cart_url(page_url),
            "cart_keyword_match": any(keyword in result_lower for keyword in self.CART_KEYWORDS),
            "checkout_keyword_match": any(
                keyword in result_lower for keyword in self.CHECKOUT_KEYWORDS
            ),
            "is_add_to_cart_action": self._contains_any(
                combined_text, self.ADD_TO_CART_ACTION_KEYWORDS
            ),
            "is_checkout_action": self._contains_any(
                combined_text, self.CHECKOUT_ACTION_KEYWORDS
            ),
            "is_form_fill": is_form_fill,
            "is_real_fill": is_real_fill,
            "is_payment_fill": is_payment_fill,
            "has_payment_fields": has_payment_fields,
            "has_payment_fields_strong": has_payment_fields_strong,
        }

    def summarize(
        self, task: AgentTask, step_signals: list[dict[str, bool]]
    ) -> dict[str, int | bool | str | dict[str, str]]:
        form_fill_count = sum(1 for signals in step_signals if signals["is_form_fill"])
        real_fill_count = sum(1 for signals in step_signals if signals["is_real_fill"])
        payment_fill_count = sum(1 for signals in step_signals if signals["is_payment_fill"])
        url_has_terminal = any(signals["is_payment_page"] for signals in step_signals)
        payment_fields_confirmed = any(
            signals["has_payment_fields_strong"] or signals["is_payment_fill"]
            for signals in step_signals
        )
        milestone_confidence: dict[str, str] = {
            "cart": "none",
            "checkout": "none",
            "payment": "none",
        }
        reached_cart = False
        reached_checkout = False
        reached_payment = False
        saw_add_to_cart_action = False
        saw_real_fill = False

        for signals in step_signals:
            if signals["cart_keyword_match"]:
                self._update_confidence(milestone_confidence, "cart", "keyword_match")
            if signals["checkout_keyword_match"]:
                self._update_confidence(milestone_confidence, "checkout", "keyword_match")
            if signals["has_payment_fields"]:
                self._update_confidence(milestone_confidence, "payment", "keyword_match")

            if signals["is_cart_page"]:
                cart_confidence = (
                    "action_sequence"
                    if signals["is_add_to_cart_action"] or saw_add_to_cart_action
                    else "url_match"
                )
                self._update_confidence(milestone_confidence, "cart", cart_confidence)
                reached_cart = True
            elif signals["cart_keyword_match"] and (
                signals["is_add_to_cart_action"] or saw_add_to_cart_action
            ):
                self._update_confidence(milestone_confidence, "cart", "action_sequence")
                reached_cart = True

            if signals["is_checkout_page"]:
                checkout_confidence = (
                    "action_sequence"
                    if signals["is_checkout_action"] or reached_cart or saw_real_fill
                    else "url_match"
                )
                self._update_confidence(
                    milestone_confidence,
                    "checkout",
                    checkout_confidence,
                )
                reached_checkout = True

            if signals["is_payment_fill"] or signals["has_payment_fields_strong"]:
                self._update_confidence(milestone_confidence, "payment", "action_sequence")
                reached_payment = True
            elif signals["is_payment_page"]:
                payment_confidence = "action_sequence" if reached_checkout else "url_match"
                self._update_confidence(milestone_confidence, "payment", payment_confidence)
                reached_payment = True

            saw_add_to_cart_action = saw_add_to_cart_action or signals["is_add_to_cart_action"]
            saw_real_fill = saw_real_fill or signals["is_real_fill"]

        step_count = len(step_signals)
        suspiciously_short = step_count <= 2 and reached_payment
        return {
            "reached_payment": reached_payment,
            "reached_checkout": reached_checkout,
            "reached_cart": reached_cart,
            "reached_payment_confidence": milestone_confidence["payment"],
            "reached_checkout_confidence": milestone_confidence["checkout"],
            "reached_cart_confidence": milestone_confidence["cart"],
            "milestone_confidence": milestone_confidence,
            "form_fill_count": form_fill_count,
            "payment_fill_count": payment_fill_count,
            "fill_count": form_fill_count,
            "real_fill_count": real_fill_count,
            "payment_fields_confirmed": payment_fields_confirmed,
            "url_has_terminal": url_has_terminal,
            "suspiciously_short": suspiciously_short,
        }

    def _update_confidence(
        self,
        confidence_by_milestone: dict[str, str],
        milestone: str,
        candidate: str,
    ) -> None:
        current = confidence_by_milestone[milestone]
        if self.CONFIDENCE_RANK[candidate] > self.CONFIDENCE_RANK[current]:
            confidence_by_milestone[milestone] = candidate

    def _is_payment_url(self, url: str | None) -> bool:
        return self._path_matches(url, self.PAYMENT_URL_PATTERNS)

    def _is_checkout_url(self, url: str | None) -> bool:
        return self._path_matches(url, self.CHECKOUT_URL_PATTERNS)

    def _is_cart_url(self, url: str | None) -> bool:
        return self._path_matches(url, self.CART_URL_PATTERNS)

    def _path_matches(self, url: str | None, patterns: list[str]) -> bool:
        path = self._url_path(url)
        return any(re.search(pattern, path) for pattern in patterns)

    def _contains_any(self, text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

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
