import re
from typing import Callable , Any


def limit_links(text: str , max_links: int = 5) -> str:
    """
    Reduces the number of links in a block of text to a defined maximum.
    Preserves the original formatting and associated summaries.
    """
    # Find all HTTP/HTTPS links
    links = re.findall ( r'(https?://[^\s]+)' , text )
    if len ( links ) <= max_links:
        return text

    # Split the text by lines
    blocks = text.split ( '\n' )
    new_blocks = [ ]
    link_counter = 0

    # Get the specific links we want to keep
    links_to_keep = set ( links[ :max_links ] )

    for line in blocks:
        # Check if the current line contains one of the links we want to keep
        line_contains_link_to_keep = any ( link in line for link in links_to_keep )

        if line_contains_link_to_keep and link_counter < max_links:
            # If the line contains a link we are keeping and we haven't hit the limit,
            # add the line and increment the counter for the link it contains.
            new_blocks.append ( line )
            link_counter += 1
            # Note: This logic assumes one link per relevant block/line.
            # If a block contains multiple links, the counter might be inaccurate
            # based on the intent (limit blocks vs limit URL count).
            # The current implementation prioritizes limiting the blocks/lines *containing* the first N links.
        elif not line_contains_link_to_keep:
            # Always keep non-link-containing lines (e.g., headers, intros, separators)
            new_blocks.append ( line )

        # Stop processing if the maximum number of link-containing blocks is reached
        if link_counter >= max_links:
            # Optionally, you might want to break immediately after processing the last allowed link block
            # For simplicity, we break on the next iteration.
            pass

    return '\n'.join ( new_blocks )


def wrap_with_limiter(func: Callable[ ... , Any ] , max_links: int = 5) -> Callable[ ... , Any ]:
    """
    Decorator that wraps a function's result. If the result is a string,
    it limits the number of links in that string using limit_links.
    """

    def wrapper(*args , **kwargs) -> Any:
        # Execute the original function
        result = func ( *args , **kwargs )

        # Apply the link limitation if the result is a string
        if isinstance ( result , str ):
            return limit_links ( result , max_links = max_links )

        # Return the original result otherwise
        return result

    return wrapper