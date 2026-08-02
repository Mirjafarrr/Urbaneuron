import { createFileRoute } from "@tanstack/react-router";
import { Urbaneuron } from "@/components/urbaneuron/Urbaneuron";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  return <Urbaneuron />;
}
