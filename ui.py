from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass
class MenuState:
    options: list[str]
    selected_index: int = 0

    def move(self, delta: int) -> bool:
        old = self.selected_index
        self.selected_index = (self.selected_index + delta) % len(self.options)
        return old != self.selected_index

    @property
    def selected(self) -> str:
        return self.options[self.selected_index]


def draw_centered_menu(
    screen: pygame.Surface,
    title: str,
    state: MenuState,
    font_title: pygame.font.Font,
    font_item: pygame.font.Font,
) -> list[pygame.Rect]:
    width, height = screen.get_size()
    # Solid fill keeps render cost low and preserves high-contrast readability.
    screen.fill((0, 0, 0))

    frame = pygame.Rect(40, 36, width - 80, height - 72)
    pygame.draw.rect(screen, (245, 245, 245), frame, 2)

    title_surface = font_title.render(title, True, (255, 255, 255))
    title_rect = title_surface.get_rect(center=(width // 2, height // 7))
    screen.blit(title_surface, title_rect)
    pygame.draw.line(screen, (210, 210, 210), (120, title_rect.bottom + 12), (width - 120, title_rect.bottom + 12), 2)

    rects: list[pygame.Rect] = [pygame.Rect(0, 0, 0, 0) for _ in state.options]
    top = title_rect.bottom + 34
    bottom = height - 64
    step = max(50, int(font_item.get_height() * 1.5))
    visible_count = max(1, (bottom - top) // step)
    start = 0
    total = len(state.options)
    if total > visible_count:
        start = max(0, min(state.selected_index - (visible_count // 2), total - visible_count))
    end = min(total, start + visible_count)

    for row, i in enumerate(range(start, end)):
        selected = i == state.selected_index
        surf = font_item.render(state.options[i], True, (255, 255, 255) if selected else (200, 200, 200))
        rect = surf.get_rect(center=(width // 2, top + row * step))
        if selected:
            highlight = pygame.Rect(140, rect.top - 8, width - 280, rect.height + 16)
            pygame.draw.rect(screen, (255, 255, 255), highlight, 2)
            pointer = font_item.render(">>", True, (255, 255, 255))
            screen.blit(pointer, pointer.get_rect(midright=(highlight.left - 16, rect.centery)))
        screen.blit(surf, rect)
        rects[i] = rect

    if start > 0:
        up = font_item.render("^ more", True, (180, 180, 180))
        screen.blit(up, up.get_rect(midtop=(width // 2, top - 28)))
    if end < total:
        down = font_item.render("v more", True, (180, 180, 180))
        screen.blit(down, down.get_rect(midbottom=(width // 2, bottom + 28)))
    return rects
