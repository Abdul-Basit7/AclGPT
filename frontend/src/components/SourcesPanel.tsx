import { FileText, PanelRightClose, Quote } from "lucide-react";

import type { Source } from "@/api/types";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Separator } from "@/components/ui/separator";
import { LG_QUERY, useMediaQuery } from "@/hooks/use-media-query";

interface Props {
  open: boolean;
  sources: Source[];
  /** Which answer the citations belong to, shown as context. */
  contextLabel: string | null;
  onClose: () => void;
}

function SourcesList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) {
    return (
      <div className="flex flex-col items-center py-10 text-center">
        <div className="bg-muted text-muted-foreground mb-3 flex size-10 items-center justify-center rounded-lg">
          <FileText className="size-4" />
        </div>
        <p className="text-sm font-medium">No citations yet</p>
        <p className="text-muted-foreground mt-1 text-sm">
          Ask a question and the passages behind the answer appear here.
        </p>
      </div>
    );
  }

  return (
    <ol className="space-y-3">
      {sources.map((source, index) => (
        <li key={`${source.filename}-${source.page}-${index}`}>
          <div className="flex items-baseline gap-2">
            <span className="bg-muted text-muted-foreground flex size-5 shrink-0 items-center justify-center rounded text-[11px] font-medium">
              {index + 1}
            </span>
            <span className="min-w-0 flex-1 truncate text-sm font-medium">
              {source.filename}
            </span>
            {source.page !== null ? (
              <span className="text-muted-foreground shrink-0 text-xs">
                p. {source.page}
              </span>
            ) : null}
          </div>
          {source.snippet ? (
            <p className="text-muted-foreground mt-1.5 pl-7 text-xs leading-relaxed">
              {source.snippet}
            </p>
          ) : null}
          {index < sources.length - 1 ? <Separator className="mt-3" /> : null}
        </li>
      ))}
    </ol>
  );
}

/**
 * Citations dock to the right on wide screens rather than sitting inline under
 * each answer: they are reference material, so they stay put while you read and
 * scroll instead of pushing the conversation around. Below `lg` the same content
 * arrives as a right-hand sheet, since there is no room to dock it.
 */
export function SourcesPanel({ open, sources, contextLabel, onClose }: Props) {
  // Branch in JS, not CSS: the sheet's overlay would otherwise stay mounted on
  // wide screens and intercept every click behind an invisible layer.
  const canDock = useMediaQuery(LG_QUERY);

  if (canDock) {
    if (!open) return null;
    return (
      <aside className="bg-sidebar flex w-80 shrink-0 flex-col border-l xl:w-96">
        <header className="flex h-14 items-center gap-2 border-b px-4">
          <Quote className="text-muted-foreground size-4 shrink-0" />
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold">Sources</h2>
            {contextLabel ? (
              <p className="text-muted-foreground truncate text-xs">
                {contextLabel}
              </p>
            ) : null}
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close sources"
          >
            <PanelRightClose />
          </Button>
        </header>

        <ScrollArea className="min-h-0 flex-1">
          <div className="p-4">
            <SourcesList sources={sources} />
          </div>
        </ScrollArea>
      </aside>
    );
  }

  // Narrow screens: same content, as a right-hand sheet.
  return (
    <Sheet
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
    >
      <SheetContent className="flex w-full flex-col gap-0 sm:max-w-sm">
        <SheetHeader>
          <SheetTitle>Sources</SheetTitle>
          {contextLabel ? (
            <SheetDescription className="truncate">
              {contextLabel}
            </SheetDescription>
          ) : null}
        </SheetHeader>
        <ScrollArea className="min-h-0 flex-1">
          <div className="p-4">
            <SourcesList sources={sources} />
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
