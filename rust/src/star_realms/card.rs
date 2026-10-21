//! The cards the Rust engine plays: the starting-deck cards, the Explorer and
//! the first slice's Trade Deck (docs/star-realms.md, "Rules and scope").
//!
//! The data mirrors `cards.toml`; a differential test checks it against the
//! Python catalogue, card by card. Each later slice adds its cards here and
//! its kinds of effect to `Effect`.

/// A card, by name. `Copy` and one byte, so piles are plain vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Card {
    Scout,
    Viper,
    Explorer,
    BlobFighter,
    TradePod,
    Ram,
    BattleBlob,
    Mothership,
    BlobWheel,
    TheHive,
    FederationShuttle,
    Cutter,
    TradeEscort,
    Flagship,
    Corvette,
    Dreadnaught,
    SpaceStation,
    WarWorld,
    BattleStation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Faction {
    Blob,
    TradeFederation,
    StarEmpire,
    MachineCult,
}

impl Faction {
    /// Its place in `GameState::factions_played`, in the Python `Faction`
    /// enum's order.
    pub fn index(self) -> usize {
        match self {
            Faction::Blob => 0,
            Faction::TradeFederation => 1,
            Faction::StarEmpire => 2,
            Faction::MachineCult => 3,
        }
    }

    /// The Python `Faction` member's value.
    pub fn name(self) -> &'static str {
        match self {
            Faction::Blob => "Blob",
            Faction::TradeFederation => "Trade Federation",
            Faction::StarEmpire => "Star Empire",
            Faction::MachineCult => "Machine Cult",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardType {
    Ship,
    Base,
    Outpost,
}

impl CardType {
    /// The Python `CardType` member's value.
    pub fn name(self) -> &'static str {
        match self {
            CardType::Ship => "ship",
            CardType::Base => "base",
            CardType::Outpost => "outpost",
        }
    }
}

/// What a card does: the first slice's kinds of effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    Trade(i32),
    Combat(i32),
    Authority(i32),
    Draw(usize),
}

impl Effect {
    /// Whether the effect only adds trade, combat or Authority
    /// (`rules._pure_gain`): a draw can reshuffle the discard pile, so it never
    /// is.
    pub fn pure_gain(self) -> bool {
        match self {
            Effect::Trade(_) | Effect::Combat(_) | Effect::Authority(_) => true,
            Effect::Draw(_) => false,
        }
    }
}

/// One card's game data, as `card.Card` holds it.
pub struct CardData {
    pub name: &'static str,
    pub cost: i32,
    pub faction: Option<Faction>,
    pub card_type: CardType,
    pub defense: i32,
    pub primary: &'static [Effect],
    pub ally: &'static [Effect],
    pub scrap: &'static [Effect],
    pub copies: usize,
}

use Effect::{Authority, Combat, Draw, Trade};

const fn card(
    name: &'static str,
    cost: i32,
    faction: Option<Faction>,
    card_type: CardType,
    defense: i32,
    copies: usize,
    effects: [&'static [Effect]; 3],
) -> CardData {
    let [primary, ally, scrap] = effects;
    CardData {
        name,
        cost,
        faction,
        card_type,
        defense,
        primary,
        ally,
        scrap,
        copies,
    }
}

const BLOB: Option<Faction> = Some(Faction::Blob);
const FEDERATION: Option<Faction> = Some(Faction::TradeFederation);
const EMPIRE: Option<Faction> = Some(Faction::StarEmpire);
const MACHINE: Option<Faction> = Some(Faction::MachineCult);
const SHIP: CardType = CardType::Ship;
const BASE: CardType = CardType::Base;
const OUTPOST: CardType = CardType::Outpost;

impl Card {
    /// Every card, in catalogue (`cards.toml`) order.
    pub const ALL: [Card; 19] = [
        Card::Scout,
        Card::Viper,
        Card::Explorer,
        Card::BlobFighter,
        Card::TradePod,
        Card::Ram,
        Card::BattleBlob,
        Card::Mothership,
        Card::BlobWheel,
        Card::TheHive,
        Card::FederationShuttle,
        Card::Cutter,
        Card::TradeEscort,
        Card::Flagship,
        Card::Corvette,
        Card::Dreadnaught,
        Card::SpaceStation,
        Card::WarWorld,
        Card::BattleStation,
    ];

    /// The card's data. An exhaustive `match`, so a new card can't be left
    /// without any.
    pub fn data(self) -> &'static CardData {
        const SCOUT: CardData = card("Scout", 0, None, SHIP, 0, 16, [&[Trade(1)], &[], &[]]);
        const VIPER: CardData = card("Viper", 0, None, SHIP, 0, 4, [&[Combat(1)], &[], &[]]);
        const EXPLORER: CardData = card(
            "Explorer",
            2,
            None,
            SHIP,
            0,
            10,
            [&[Trade(2)], &[], &[Combat(2)]],
        );
        const BLOB_FIGHTER: CardData = card(
            "Blob Fighter",
            1,
            BLOB,
            SHIP,
            0,
            3,
            [&[Combat(3)], &[Draw(1)], &[]],
        );
        const TRADE_POD: CardData = card(
            "Trade Pod",
            2,
            BLOB,
            SHIP,
            0,
            3,
            [&[Trade(3)], &[Combat(2)], &[]],
        );
        const RAM: CardData = card(
            "Ram",
            3,
            BLOB,
            SHIP,
            0,
            2,
            [&[Combat(5)], &[Combat(2)], &[Trade(3)]],
        );
        const BATTLE_BLOB: CardData = card(
            "Battle Blob",
            6,
            BLOB,
            SHIP,
            0,
            1,
            [&[Combat(8)], &[Draw(1)], &[Combat(4)]],
        );
        const MOTHERSHIP: CardData = card(
            "Mothership",
            7,
            BLOB,
            SHIP,
            0,
            1,
            [&[Combat(6), Draw(1)], &[Draw(1)], &[]],
        );
        const BLOB_WHEEL: CardData = card(
            "Blob Wheel",
            3,
            BLOB,
            BASE,
            5,
            3,
            [&[Combat(1)], &[], &[Trade(3)]],
        );
        const THE_HIVE: CardData = card(
            "The Hive",
            5,
            BLOB,
            BASE,
            5,
            1,
            [&[Combat(3)], &[Draw(1)], &[]],
        );
        const FEDERATION_SHUTTLE: CardData = card(
            "Federation Shuttle",
            1,
            FEDERATION,
            SHIP,
            0,
            3,
            [&[Trade(2)], &[Authority(4)], &[]],
        );
        const CUTTER: CardData = card(
            "Cutter",
            2,
            FEDERATION,
            SHIP,
            0,
            3,
            [&[Authority(4), Trade(2)], &[Combat(4)], &[]],
        );
        const TRADE_ESCORT: CardData = card(
            "Trade Escort",
            5,
            FEDERATION,
            SHIP,
            0,
            1,
            [&[Authority(4), Combat(4)], &[Draw(1)], &[]],
        );
        const FLAGSHIP: CardData = card(
            "Flagship",
            6,
            FEDERATION,
            SHIP,
            0,
            1,
            [&[Combat(5), Draw(1)], &[Authority(5)], &[]],
        );
        const CORVETTE: CardData = card(
            "Corvette",
            2,
            EMPIRE,
            SHIP,
            0,
            2,
            [&[Combat(1), Draw(1)], &[Combat(2)], &[]],
        );
        const DREADNAUGHT: CardData = card(
            "Dreadnaught",
            7,
            EMPIRE,
            SHIP,
            0,
            1,
            [&[Combat(7), Draw(1)], &[], &[Combat(5)]],
        );
        const SPACE_STATION: CardData = card(
            "Space Station",
            4,
            EMPIRE,
            OUTPOST,
            4,
            2,
            [&[Combat(2)], &[Combat(2)], &[Trade(4)]],
        );
        const WAR_WORLD: CardData = card(
            "War World",
            5,
            EMPIRE,
            OUTPOST,
            4,
            1,
            [&[Combat(3)], &[Combat(4)], &[]],
        );
        const BATTLE_STATION: CardData = card(
            "Battle Station",
            3,
            MACHINE,
            OUTPOST,
            5,
            2,
            [&[], &[], &[Combat(5)]],
        );
        match self {
            Card::Scout => &SCOUT,
            Card::Viper => &VIPER,
            Card::Explorer => &EXPLORER,
            Card::BlobFighter => &BLOB_FIGHTER,
            Card::TradePod => &TRADE_POD,
            Card::Ram => &RAM,
            Card::BattleBlob => &BATTLE_BLOB,
            Card::Mothership => &MOTHERSHIP,
            Card::BlobWheel => &BLOB_WHEEL,
            Card::TheHive => &THE_HIVE,
            Card::FederationShuttle => &FEDERATION_SHUTTLE,
            Card::Cutter => &CUTTER,
            Card::TradeEscort => &TRADE_ESCORT,
            Card::Flagship => &FLAGSHIP,
            Card::Corvette => &CORVETTE,
            Card::Dreadnaught => &DREADNAUGHT,
            Card::SpaceStation => &SPACE_STATION,
            Card::WarWorld => &WAR_WORLD,
            Card::BattleStation => &BATTLE_STATION,
        }
    }

    pub fn name(self) -> &'static str {
        self.data().name
    }

    pub fn from_name(name: &str) -> Option<Card> {
        Card::ALL.into_iter().find(|card| card.name() == name)
    }

    pub fn is_ship(self) -> bool {
        self.data().card_type == CardType::Ship
    }

    pub fn is_outpost(self) -> bool {
        self.data().card_type == CardType::Outpost
    }
}

/// Each player's starting deck: 8 Scouts and 2 Vipers.
pub const STARTING_DECK: [Card; 10] = [
    Card::Scout,
    Card::Scout,
    Card::Scout,
    Card::Scout,
    Card::Scout,
    Card::Scout,
    Card::Scout,
    Card::Scout,
    Card::Viper,
    Card::Viper,
];

/// How many Explorers the game starts with.
pub const EXPLORER_PILE: u32 = 10;

/// The first slice's Trade Deck: every faction card, each repeated `copies`
/// times, in catalogue order (`catalogue.playable_deck`), so a seed shuffles
/// it as Python does.
pub fn trade_deck() -> Vec<Card> {
    Card::ALL
        .into_iter()
        .filter(|card| card.data().faction.is_some())
        .flat_map(|card| std::iter::repeat_n(card, card.data().copies))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_slice_has_30_cards_9_of_them_bases() {
        let deck = trade_deck();
        assert_eq!(deck.len(), 30);
        assert_eq!(deck.iter().filter(|card| !card.is_ship()).count(), 9);
    }

    #[test]
    fn names_round_trip() {
        for card in Card::ALL {
            assert_eq!(Card::from_name(card.name()), Some(card));
        }
        assert_eq!(Card::from_name("Stealth Needle"), None);
    }
}
