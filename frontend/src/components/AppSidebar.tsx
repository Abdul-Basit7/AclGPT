import { useState } from "react";
import {
  Check,
  ChevronsUpDown,
  FolderOpen,
  LogOut,
  MessageSquare,
  Plus,
  Trash2,
} from "lucide-react";

import type { Chat, Collection } from "@/api/types";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Logo } from "@/components/logo";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInput,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";

interface Props {
  email: string;
  collections: Collection[];
  activeCollectionId: number | null;
  chats: Chat[];
  activeChatId: number | null;
  onSelectCollection: (id: number) => void;
  onCreateCollection: (name: string) => void;
  onDeleteCollection: (id: number) => void;
  onNewChat: () => void;
  onSelectChat: (id: number) => void;
  onDeleteChat: (id: number) => void;
  onLogout: () => void;
}

export function AppSidebar({
  email,
  collections,
  activeCollectionId,
  chats,
  activeChatId,
  onSelectCollection,
  onCreateCollection,
  onDeleteCollection,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onLogout,
}: Props) {
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");

  const activeCollection = collections.find((c) => c.id === activeCollectionId);
  const visibleChats = chats.filter((c) => c.collection_id === activeCollectionId);

  function submitCollection(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    onCreateCollection(trimmed);
    setName("");
    setCreating(false);
  }

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" className="cursor-default hover:bg-transparent">
              <div className="bg-primary text-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg">
                <Logo className="size-4" />
              </div>
              <div className="grid flex-1 text-left leading-tight">
                <span className="truncate font-semibold">Sourcery</span>
                <span className="text-muted-foreground truncate text-xs">
                  Answers with receipts
                </span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Collection</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <SidebarMenuButton tooltip={activeCollection?.name ?? "Collections"}>
                      <FolderOpen />
                      <span className="truncate">
                        {activeCollection?.name ?? "No collection"}
                      </span>
                      <ChevronsUpDown className="ml-auto" />
                    </SidebarMenuButton>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="w-56">
                    <DropdownMenuLabel className="text-muted-foreground text-xs">
                      Collections
                    </DropdownMenuLabel>
                    {collections.map((collection) => (
                      <DropdownMenuItem
                        key={collection.id}
                        onClick={() => onSelectCollection(collection.id)}
                      >
                        <FolderOpen />
                        <span className="truncate">{collection.name}</span>
                        <span className="text-muted-foreground ml-auto text-xs">
                          {collection.ready_count}/{collection.document_count}
                        </span>
                        {collection.id === activeCollectionId ? (
                          <Check className="size-3.5" />
                        ) : null}
                      </DropdownMenuItem>
                    ))}
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => setCreating(true)}>
                      <Plus />
                      New collection
                    </DropdownMenuItem>
                    {activeCollection && collections.length > 1 ? (
                      <ConfirmDialog
                        title={`Delete “${activeCollection.name}”?`}
                        description="Its documents, indexes and chats are removed permanently. This cannot be undone."
                        confirmLabel="Delete collection"
                        onConfirm={() => onDeleteCollection(activeCollection.id)}
                        trigger={
                          <DropdownMenuItem
                            variant="destructive"
                            onSelect={(e) => e.preventDefault()}
                          >
                            <Trash2 />
                            Delete collection
                          </DropdownMenuItem>
                        }
                      />
                    ) : null}
                  </DropdownMenuContent>
                </DropdownMenu>
              </SidebarMenuItem>

              {creating ? (
                <SidebarMenuItem className="group-data-[collapsible=icon]:hidden">
                  <form onSubmit={submitCollection} className="px-1 py-1">
                    <SidebarInput
                      autoFocus
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      onBlur={() => !name.trim() && setCreating(false)}
                      placeholder="Name, then Enter"
                      maxLength={120}
                    />
                  </form>
                </SidebarMenuItem>
              ) : null}

              <SidebarMenuItem>
                <SidebarMenuButton onClick={onNewChat} tooltip="New chat">
                  <Plus />
                  <span>New chat</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
          <SidebarGroupLabel>Chats</SidebarGroupLabel>
          <SidebarGroupContent>
            {visibleChats.length === 0 ? (
              <p className="text-muted-foreground px-2 py-1.5 text-xs">
                No chats in this collection yet.
              </p>
            ) : (
              <SidebarMenu>
                {visibleChats.map((chat) => (
                  <SidebarMenuItem key={chat.id}>
                    <SidebarMenuButton
                      isActive={chat.id === activeChatId}
                      onClick={() => onSelectChat(chat.id)}
                      tooltip={chat.title || "New chat"}
                    >
                      <MessageSquare />
                      <span className="truncate">{chat.title || "New chat"}</span>
                    </SidebarMenuButton>
                    <ConfirmDialog
                      title="Delete this chat?"
                      description={`“${chat.title || "New chat"}” and its messages are removed permanently.`}
                      confirmLabel="Delete chat"
                      onConfirm={() => onDeleteChat(chat.id)}
                      trigger={
                        <SidebarMenuAction
                          showOnHover
                          aria-label={`Delete chat ${chat.title}`}
                        >
                          <Trash2 />
                        </SidebarMenuAction>
                      }
                    />
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton size="lg" tooltip={email}>
                  <div className="bg-muted text-muted-foreground flex aspect-square size-8 items-center justify-center rounded-lg text-xs font-medium uppercase">
                    {email.slice(0, 1)}
                  </div>
                  <div className="grid flex-1 text-left leading-tight">
                    <span className="truncate text-sm">{email}</span>
                  </div>
                  <ChevronsUpDown className="ml-auto" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="top" align="start" className="w-56">
                <DropdownMenuLabel className="truncate font-normal">
                  {email}
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={onLogout}>
                  <LogOut />
                  Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>

      {/* Drag/click edge to collapse, in addition to the header trigger. */}
      <SidebarRail />
    </Sidebar>
  );
}
