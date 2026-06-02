import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_token(name: str = "MYSTIC", symbol: str = "MYST", supply: int = 1_000_000) -> None:
    """
    Generate a minimal ERC-20-like Solidity contract file.

    Args:
        name: Token name (human-readable)
        symbol: Token symbol (contract name & ticker)
        supply: Total token supply (before decimals)

    Output:
        Writes a Solidity contract file: <symbol>_Token.sol
    """
    # Basic validation
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", symbol):
        msg = f"Invalid Solidity identifier for symbol: {symbol!r}"
        raise ValueError(msg)

    if not isinstance(supply, int):
        msg = "Supply must be an integer"
        raise TypeError(msg)

    if supply <= 0:
        msg = "Supply must be positive"
        raise ValueError(msg)

    # Escape name for inclusion in Solidity string literal
    # (escape backslashes and double quotes)
    name_escaped = name.replace("\\", "\\\\").replace('"', '\\"')

    logger.info(f"[TOKEN] Generating ERC-20 token '{name}' ({symbol}) with supply {supply}")

    contract_code = f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract {symbol} {{
    string public name = "{name_escaped}";
    string public symbol = "{symbol}";
    uint8 public constant decimals = 18;
    uint256 public totalSupply = {supply} * (10 ** uint256(decimals));
    mapping(address => uint256) public balanceOf;

    event Transfer(address indexed from, address indexed to, uint256 value);

    constructor() {{
        balanceOf[msg.sender] = totalSupply;
    }}

    function transfer(address to, uint256 value) public returns (bool success) {{
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }}
}}
"""

    file_name = Path(f"{symbol}_Token.sol")
    with file_name.open("w", encoding="utf-8") as f:
        f.write(contract_code)

    logger.info(f"[TOKEN] Contract written to {file_name}")
