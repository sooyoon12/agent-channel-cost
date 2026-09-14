"""Surface renderers fed by one language-independent semantic form."""

from __future__ import annotations

from typing import Any, Mapping


KO_DATE = {
    "before": "이전",
    "after": "이후",
    "on_or_before": "당일을 포함한 이전",
    "on_or_after": "당일을 포함한 이후",
}
EN_DATE = {
    "before": "before",
    "after": "after",
    "on_or_before": "on or before",
    "on_or_after": "on or after",
}
KO_ATTENDANCE = {"attending": "참석하는", "not_attending": "참석하지 않는"}
EN_ATTENDANCE = {"attending": "is attending", "not_attending": "is not attending"}
KO_COUNT = {"at_least": "이상", "at_most": "이하"}
EN_COUNT = {"at_least": "at least", "at_most": "at most"}
KO_SORT = {"ascending": "오름차순", "descending": "내림차순"}
EN_SORT = {"ascending": "ascending", "descending": "descending"}
KO_REGIONS = {"EAST": "동부", "WEST": "서부", "SOUTH": "남부", "NORTH": "북부"}
EN_REGIONS = {"EAST": "East", "WEST": "West", "SOUTH": "South", "NORTH": "North"}
KO_STATUSES = {"REVIEW": "검토", "HOLD": "보류", "APPROVED": "승인"}
EN_STATUSES = {"REVIEW": "review", "HOLD": "on hold", "APPROVED": "approved"}


def render_precise(semantic: Mapping[str, Any]) -> tuple[str, str]:
    action = semantic["action"]
    if action == "search":
        cancelled_ko = "취소된 일정도 포함하고" if semantic["include_cancelled"] else "취소된 일정은 제외하고"
        cancelled_en = "include cancelled events" if semantic["include_cancelled"] else "exclude cancelled events"
        ko = (
            f"일정 도구에서 {semantic['person_ko']}({semantic['person_id']})가 "
            f"{KO_ATTENDANCE[semantic['attendance']]} 일정 중 "
            f"{semantic['reference_date']} {KO_DATE[semantic['date_operator']]}이고 참석자 수가 "
            f"{semantic['attendee_count']}명 {KO_COUNT[semantic['attendee_operator']]}인 것만 찾아줘. "
            f"{cancelled_ko}, 날짜 {KO_SORT[semantic['sort']]}으로 정렬해."
        )
        en = (
            f"Using the schedule tool, find only events that {semantic['person_en']} "
            f"({semantic['person_id']}) "
            f"{EN_ATTENDANCE[semantic['attendance']]}, are {EN_DATE[semantic['date_operator']]} "
            f"{semantic['reference_date']}, and have {EN_COUNT[semantic['attendee_operator']]} "
            f"{semantic['attendee_count']} attendees. {cancelled_en.capitalize()} and sort by date "
            f"in {EN_SORT[semantic['sort']]} order."
        )
        return ko, en
    if action == "set_attendance":
        status_ko = "참석" if semantic["status"] == "attending" else "불참"
        status_en = "attending" if semantic["status"] == "attending" else "not attending"
        return (
            f"일정 {semantic['event_id']}에서 {semantic['person_ko']}({semantic['person_id']})의 상태를 "
            f"참석이 아니라 {status_ko}으로 변경해.",
            f"For event {semantic['event_id']}, change {semantic['person_en']} "
            f"({semantic['person_id']}) to {status_en}; do not set the opposite status.",
        )
    if action == "delete":
        audit_ko = "감사 기록은 보존해" if semantic["preserve_audit"] else "감사 기록도 보존하지 마"
        audit_en = "preserve its audit record" if semantic["preserve_audit"] else "do not preserve its audit record"
        return (
            f"일정 {semantic['event_id']}을 삭제하되, {audit_ko}.",
            f"Delete event {semantic['event_id']} and {audit_en}.",
        )
    raise ValueError(f"unknown precise action: {action}")


def render_long_observation(semantic: Mapping[str, Any]) -> tuple[str, str]:
    if "minimum_amount" not in semantic:
        ko = (
            f"데이터셋 {semantic['dataset_id']}의 전체 레코드를 도구로 읽어. 그중 지역이 "
            f"{KO_REGIONS[semantic['region']]}({semantic['region']})이고 상태가 "
            f"{KO_STATUSES[semantic['status']]}({semantic['status']})인 레코드가 정확히 하나 있다. "
            "그 레코드의 ID와 레코드 안의 evidence_code를 submit_evidence로 제출해."
        )
        en = (
            f"Read every record in dataset {semantic['dataset_id']} with the tool. Exactly one record has "
            f"region {EN_REGIONS[semantic['region']]} ({semantic['region']}) and status "
            f"{EN_STATUSES[semantic['status']]} ({semantic['status']}). Submit that record's ID and its own "
            "evidence_code with submit_evidence."
        )
        return ko, en
    ko = (
        f"데이터셋 {semantic['dataset_id']}의 전체 레코드를 도구로 읽어. 그중 지역이 "
        f"{KO_REGIONS[semantic['region']]}({semantic['region']})이고, 상태가 "
        f"{KO_STATUSES[semantic['status']]}({semantic['status']})이며, 금액이 "
        f"{semantic['minimum_amount']} 이상이고 날짜가 "
        f"{semantic['on_or_after']} 당일을 포함해 이후인 레코드가 정확히 하나 있다. 그 레코드의 ID와 "
        "레코드 안의 evidence_code를 submit_evidence로 제출해. 비슷한 레코드를 추측해서는 안 돼."
    )
    en = (
        f"Read every record in dataset {semantic['dataset_id']} with the tool. Exactly one record has region "
        f"{EN_REGIONS[semantic['region']]} ({semantic['region']}), status "
        f"{EN_STATUSES[semantic['status']]} ({semantic['status']}), amount at least "
        f"{semantic['minimum_amount']}, and "
        f"a date on or after {semantic['on_or_after']}. Submit that record's ID and its own evidence_code "
        "with submit_evidence. Do not guess from similar records."
    )
    return ko, en
