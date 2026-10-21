//! The cards the Rust engine plays: the base set's 80-card Trade Deck, the
//! Explorer and the starting-deck cards (docs/star-realms.md, "Rules and
//! scope"), and the copies Stealth Needle makes of ships in play.
//!
//! The data mirrors `cards.toml`; a differential test checks it against the
//! Python catalogue, card by card.

use std::sync::OnceLock;

/// A card, by name, or Stealth Needle as a copy of another ship
/// (`NeedleAs`, `rules._copy`). `Copy` and two bytes, so piles are plain
/// lists of plain data (`state::Pile`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Card {
    Scout,
    Viper,
    Explorer,
    BlobFighter,
    BattlePod,
    TradePod,
    Ram,
    BlobDestroyer,
    BattleBlob,
    BlobCarrier,
    Mothership,
    BlobWheel,
    TheHive,
    BlobWorld,
    FederationShuttle,
    Cutter,
    EmbassyYacht,
    Freighter,
    TradeEscort,
    Flagship,
    CommandShip,
    TradingPost,
    BarterWorld,
    DefenseCenter,
    PortOfCall,
    CentralOffice,
    ImperialFighter,
    Corvette,
    ImperialFrigate,
    SurveyShip,
    Battlecruiser,
    Dreadnaught,
    SpaceStation,
    RecyclingStation,
    WarWorld,
    RoyalRedoubt,
    FleetHq,
    TradeBot,
    MissileBot,
    SupplyBot,
    PatrolMech,
    StealthNeedle,
    BattleMech,
    MissileMech,
    BattleStation,
    MechWorld,
    Junkyard,
    MachineBase,
    BrainWorld,
    /// Stealth Needle in play as a copy of the ship `Card::ALL[i]`: a ship of
    /// its own, `Stealth Needle (Ram)` say, with the target's abilities and
    /// faction, which turns back into Stealth Needle when it leaves play
    /// (`Card::original`). Made only by `Card::needle_as`.
    NeedleAs(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Faction {
    Blob,
    TradeFederation,
    StarEmpire,
    MachineCult,
}

impl Faction {
    /// Every faction, in the Python `Faction` enum's order.
    pub const ALL: [Faction; 4] = [
        Faction::Blob,
        Faction::TradeFederation,
        Faction::StarEmpire,
        Faction::MachineCult,
    ];

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

/// What a card does: `effect.Effect`, every variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    Trade(i32),
    Combat(i32),
    Authority(i32),
    Draw(usize),
    /// Target opponent discards this many cards.
    OpponentDiscards(u32),
    /// You may destroy target base.
    DestroyBase,
    /// You may scrap up to this many cards from hand and/or discard pile.
    ScrapHandOrDiscard(u32),
    /// You may scrap up to this many cards in the Trade Row.
    ScrapTradeRow(u32),
    /// You may put the next ship you acquire this turn on top of your deck.
    NextShipOnTop,
    /// Do exactly one of the options, each a group of effects.
    Choice(&'static [&'static [Effect]]),
    /// Acquire any ship in the Trade Row for free, on top of your deck.
    AcquireShipFreeOnTop,
    /// With at least `bases` bases in play, draw `cards` (Embassy Yacht).
    DrawIfBases {
        bases: usize,
        cards: usize,
    },
    /// Draw a card per card of this card's faction played this turn.
    DrawPerFactionCard,
    /// Discard up to this many cards, then draw that many.
    DiscardThenDraw(u32),
    /// Scrap up to this many cards from hand and/or discard pile, then draw
    /// that many (Brain World).
    ScrapThenDraw(u32),
    /// Scrap a card from your hand: not optional (Machine Base).
    ScrapFromHand,
    /// Copy another ship in play (Stealth Needle).
    CopyShip,
    /// Every ship played gains this much combat (Fleet HQ), always on.
    ShipsGainCombat(i32),
    /// This card counts as an ally for every faction (Mech World), always on.
    AllyForAllFactions,
}

impl Effect {
    /// The Python variant's class name.
    pub fn kind(self) -> &'static str {
        match self {
            Effect::Trade(_) => "Trade",
            Effect::Combat(_) => "Combat",
            Effect::Authority(_) => "Authority",
            Effect::Draw(_) => "Draw",
            Effect::OpponentDiscards(_) => "OpponentDiscards",
            Effect::DestroyBase => "DestroyBase",
            Effect::ScrapHandOrDiscard(_) => "ScrapHandOrDiscard",
            Effect::ScrapTradeRow(_) => "ScrapTradeRow",
            Effect::NextShipOnTop => "NextShipOnTop",
            Effect::Choice(_) => "Choice",
            Effect::AcquireShipFreeOnTop => "AcquireShipFreeOnTop",
            Effect::DrawIfBases { .. } => "DrawIfBases",
            Effect::DrawPerFactionCard => "DrawPerFactionCard",
            Effect::DiscardThenDraw(_) => "DiscardThenDraw",
            Effect::ScrapThenDraw(_) => "ScrapThenDraw",
            Effect::ScrapFromHand => "ScrapFromHand",
            Effect::CopyShip => "CopyShip",
            Effect::ShipsGainCombat(_) => "ShipsGainCombat",
            Effect::AllyForAllFactions => "AllyForAllFactions",
        }
    }

    /// Whether the effect is pure gain (`rules._pure_gain`): it only adds
    /// trade, combat or Authority, makes the opponent discard, lets the player
    /// scrap from hand or discard pile or from the Trade Row, destroy a base
    /// or put a ship they acquire on top of their deck, or is an "or" choice
    /// between options that do only that.
    pub fn pure_gain(self) -> bool {
        match self {
            Effect::Trade(_)
            | Effect::Combat(_)
            | Effect::Authority(_)
            | Effect::OpponentDiscards(_)
            | Effect::ScrapHandOrDiscard(_)
            | Effect::DestroyBase
            | Effect::ScrapTradeRow(_)
            | Effect::NextShipOnTop => true,
            Effect::Choice(options) => options.iter().all(|option| pure_gain(option)),
            Effect::Draw(_)
            | Effect::AcquireShipFreeOnTop
            | Effect::DrawIfBases { .. }
            | Effect::DrawPerFactionCard
            | Effect::DiscardThenDraw(_)
            | Effect::ScrapThenDraw(_)
            | Effect::ScrapFromHand
            | Effect::CopyShip
            | Effect::ShipsGainCombat(_)
            | Effect::AllyForAllFactions => false,
        }
    }

    /// Whether the effect scraps a card from hand (`rules._scraps_hand`).
    pub fn scraps_hand(self) -> bool {
        match self {
            Effect::ScrapHandOrDiscard(_) | Effect::ScrapFromHand | Effect::ScrapThenDraw(_) => {
                true
            }
            Effect::Trade(_)
            | Effect::Combat(_)
            | Effect::Authority(_)
            | Effect::Draw(_)
            | Effect::OpponentDiscards(_)
            | Effect::DestroyBase
            | Effect::ScrapTradeRow(_)
            | Effect::NextShipOnTop
            | Effect::Choice(_)
            | Effect::AcquireShipFreeOnTop
            | Effect::DrawIfBases { .. }
            | Effect::DrawPerFactionCard
            | Effect::DiscardThenDraw(_)
            | Effect::CopyShip
            | Effect::ShipsGainCombat(_)
            | Effect::AllyForAllFactions => false,
        }
    }

    /// Whether the effect works while its base is in play, without being
    /// used (`rules._ALWAYS_ON`).
    pub fn always_on(self) -> bool {
        matches!(
            self,
            Effect::ShipsGainCombat(_) | Effect::AllyForAllFactions
        )
    }
}

/// Whether every one of `effects` is pure gain (`rules._pure_gain`).
pub fn pure_gain(effects: &[Effect]) -> bool {
    effects.iter().all(|e| e.pure_gain())
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

use Effect::{
    AcquireShipFreeOnTop, AllyForAllFactions, Authority, Choice, Combat, CopyShip, DestroyBase,
    DiscardThenDraw, Draw, DrawIfBases, DrawPerFactionCard, NextShipOnTop, OpponentDiscards,
    ScrapFromHand, ScrapHandOrDiscard, ScrapThenDraw, ScrapTradeRow, ShipsGainCombat, Trade,
};

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

/// Each copy Stealth Needle can make, indexed like `Card::ALL` (built once).
fn needle_copies() -> &'static [CardData] {
    static COPIES: OnceLock<Vec<CardData>> = OnceLock::new();
    COPIES.get_or_init(|| {
        let needle = Card::StealthNeedle.data();
        Card::ALL
            .iter()
            .map(|target| {
                let data = target.data();
                let name = format!("{} ({})", needle.name, data.name);
                CardData {
                    name: Box::leak(name.into_boxed_str()),
                    cost: needle.cost,
                    faction: data.faction.or(needle.faction),
                    card_type: CardType::Ship,
                    defense: 0,
                    primary: data.primary,
                    ally: data.ally,
                    scrap: data.scrap,
                    copies: needle.copies,
                }
            })
            .collect()
    })
}

impl Card {
    /// Every card of the catalogue, in its (`cards.toml`) order: not the
    /// copies Stealth Needle makes.
    pub const ALL: [Card; 49] = [
        Card::Scout,
        Card::Viper,
        Card::Explorer,
        Card::BlobFighter,
        Card::BattlePod,
        Card::TradePod,
        Card::Ram,
        Card::BlobDestroyer,
        Card::BattleBlob,
        Card::BlobCarrier,
        Card::Mothership,
        Card::BlobWheel,
        Card::TheHive,
        Card::BlobWorld,
        Card::FederationShuttle,
        Card::Cutter,
        Card::EmbassyYacht,
        Card::Freighter,
        Card::TradeEscort,
        Card::Flagship,
        Card::CommandShip,
        Card::TradingPost,
        Card::BarterWorld,
        Card::DefenseCenter,
        Card::PortOfCall,
        Card::CentralOffice,
        Card::ImperialFighter,
        Card::Corvette,
        Card::ImperialFrigate,
        Card::SurveyShip,
        Card::Battlecruiser,
        Card::Dreadnaught,
        Card::SpaceStation,
        Card::RecyclingStation,
        Card::WarWorld,
        Card::RoyalRedoubt,
        Card::FleetHq,
        Card::TradeBot,
        Card::MissileBot,
        Card::SupplyBot,
        Card::PatrolMech,
        Card::StealthNeedle,
        Card::BattleMech,
        Card::MissileMech,
        Card::BattleStation,
        Card::MechWorld,
        Card::Junkyard,
        Card::MachineBase,
        Card::BrainWorld,
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
        // Blob
        const BLOB_FIGHTER: CardData = card(
            "Blob Fighter",
            1,
            BLOB,
            SHIP,
            0,
            3,
            [&[Combat(3)], &[Draw(1)], &[]],
        );
        const BATTLE_POD: CardData = card(
            "Battle Pod",
            2,
            BLOB,
            SHIP,
            0,
            2,
            [&[Combat(4), ScrapTradeRow(1)], &[Combat(2)], &[]],
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
        const BLOB_DESTROYER: CardData = card(
            "Blob Destroyer",
            4,
            BLOB,
            SHIP,
            0,
            2,
            [&[Combat(6)], &[DestroyBase, ScrapTradeRow(1)], &[]],
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
        const BLOB_CARRIER: CardData = card(
            "Blob Carrier",
            6,
            BLOB,
            SHIP,
            0,
            1,
            [&[Combat(7)], &[AcquireShipFreeOnTop], &[]],
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
        const BLOB_WORLD: CardData = card(
            "Blob World",
            8,
            BLOB,
            BASE,
            7,
            1,
            [&[Choice(&[&[Combat(5)], &[DrawPerFactionCard]])], &[], &[]],
        );
        // Trade Federation
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
        const EMBASSY_YACHT: CardData = card(
            "Embassy Yacht",
            3,
            FEDERATION,
            SHIP,
            0,
            2,
            [
                &[Authority(3), Trade(2), DrawIfBases { bases: 2, cards: 2 }],
                &[],
                &[],
            ],
        );
        const FREIGHTER: CardData = card(
            "Freighter",
            4,
            FEDERATION,
            SHIP,
            0,
            2,
            [&[Trade(4)], &[NextShipOnTop], &[]],
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
        const COMMAND_SHIP: CardData = card(
            "Command Ship",
            8,
            FEDERATION,
            SHIP,
            0,
            1,
            [&[Authority(4), Combat(5), Draw(2)], &[DestroyBase], &[]],
        );
        const TRADING_POST: CardData = card(
            "Trading Post",
            3,
            FEDERATION,
            OUTPOST,
            4,
            2,
            [
                &[Choice(&[&[Authority(1)], &[Trade(1)]])],
                &[],
                &[Combat(3)],
            ],
        );
        const BARTER_WORLD: CardData = card(
            "Barter World",
            4,
            FEDERATION,
            BASE,
            4,
            2,
            [
                &[Choice(&[&[Authority(2)], &[Trade(2)]])],
                &[],
                &[Combat(5)],
            ],
        );
        const DEFENSE_CENTER: CardData = card(
            "Defense Center",
            5,
            FEDERATION,
            OUTPOST,
            5,
            1,
            [
                &[Choice(&[&[Authority(3)], &[Combat(2)]])],
                &[Combat(2)],
                &[],
            ],
        );
        const PORT_OF_CALL: CardData = card(
            "Port of Call",
            6,
            FEDERATION,
            OUTPOST,
            6,
            1,
            [&[Trade(3)], &[], &[Draw(1), DestroyBase]],
        );
        const CENTRAL_OFFICE: CardData = card(
            "Central Office",
            7,
            FEDERATION,
            BASE,
            6,
            1,
            [&[Trade(2), NextShipOnTop], &[Draw(1)], &[]],
        );
        // Star Empire
        const IMPERIAL_FIGHTER: CardData = card(
            "Imperial Fighter",
            1,
            EMPIRE,
            SHIP,
            0,
            3,
            [&[Combat(2), OpponentDiscards(1)], &[Combat(2)], &[]],
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
        const IMPERIAL_FRIGATE: CardData = card(
            "Imperial Frigate",
            3,
            EMPIRE,
            SHIP,
            0,
            3,
            [&[Combat(4), OpponentDiscards(1)], &[Combat(2)], &[Draw(1)]],
        );
        const SURVEY_SHIP: CardData = card(
            "Survey Ship",
            3,
            EMPIRE,
            SHIP,
            0,
            3,
            [&[Trade(1), Draw(1)], &[], &[OpponentDiscards(1)]],
        );
        const BATTLECRUISER: CardData = card(
            "Battlecruiser",
            6,
            EMPIRE,
            SHIP,
            0,
            1,
            [
                &[Combat(5), Draw(1)],
                &[OpponentDiscards(1)],
                &[Draw(1), DestroyBase],
            ],
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
        const RECYCLING_STATION: CardData = card(
            "Recycling Station",
            4,
            EMPIRE,
            OUTPOST,
            4,
            2,
            [&[Choice(&[&[Trade(1)], &[DiscardThenDraw(2)]])], &[], &[]],
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
        const ROYAL_REDOUBT: CardData = card(
            "Royal Redoubt",
            6,
            EMPIRE,
            OUTPOST,
            6,
            1,
            [&[Combat(3)], &[OpponentDiscards(1)], &[]],
        );
        const FLEET_HQ: CardData = card(
            "Fleet HQ",
            8,
            EMPIRE,
            BASE,
            8,
            1,
            [&[ShipsGainCombat(1)], &[], &[]],
        );
        // Machine Cult
        const TRADE_BOT: CardData = card(
            "Trade Bot",
            1,
            MACHINE,
            SHIP,
            0,
            3,
            [&[Trade(1), ScrapHandOrDiscard(1)], &[Combat(2)], &[]],
        );
        const MISSILE_BOT: CardData = card(
            "Missile Bot",
            2,
            MACHINE,
            SHIP,
            0,
            3,
            [&[Combat(2), ScrapHandOrDiscard(1)], &[Combat(2)], &[]],
        );
        const SUPPLY_BOT: CardData = card(
            "Supply Bot",
            3,
            MACHINE,
            SHIP,
            0,
            3,
            [&[Trade(2), ScrapHandOrDiscard(1)], &[Combat(2)], &[]],
        );
        const PATROL_MECH: CardData = card(
            "Patrol Mech",
            4,
            MACHINE,
            SHIP,
            0,
            2,
            [
                &[Choice(&[&[Trade(3)], &[Combat(5)]])],
                &[ScrapHandOrDiscard(1)],
                &[],
            ],
        );
        const STEALTH_NEEDLE: CardData = card(
            "Stealth Needle",
            4,
            MACHINE,
            SHIP,
            0,
            1,
            [&[CopyShip], &[], &[]],
        );
        const BATTLE_MECH: CardData = card(
            "Battle Mech",
            5,
            MACHINE,
            SHIP,
            0,
            1,
            [&[Combat(4), ScrapHandOrDiscard(1)], &[Draw(1)], &[]],
        );
        const MISSILE_MECH: CardData = card(
            "Missile Mech",
            6,
            MACHINE,
            SHIP,
            0,
            1,
            [&[Combat(6), DestroyBase], &[Draw(1)], &[]],
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
        const MECH_WORLD: CardData = card(
            "Mech World",
            5,
            MACHINE,
            OUTPOST,
            6,
            1,
            [&[AllyForAllFactions], &[], &[]],
        );
        const JUNKYARD: CardData = card(
            "Junkyard",
            6,
            MACHINE,
            OUTPOST,
            5,
            1,
            [&[ScrapHandOrDiscard(1)], &[], &[]],
        );
        const MACHINE_BASE: CardData = card(
            "Machine Base",
            7,
            MACHINE,
            OUTPOST,
            6,
            1,
            [&[Draw(1), ScrapFromHand], &[], &[]],
        );
        const BRAIN_WORLD: CardData = card(
            "Brain World",
            8,
            MACHINE,
            OUTPOST,
            6,
            1,
            [&[ScrapThenDraw(2)], &[], &[]],
        );
        match self {
            Card::Scout => &SCOUT,
            Card::Viper => &VIPER,
            Card::Explorer => &EXPLORER,
            Card::BlobFighter => &BLOB_FIGHTER,
            Card::BattlePod => &BATTLE_POD,
            Card::TradePod => &TRADE_POD,
            Card::Ram => &RAM,
            Card::BlobDestroyer => &BLOB_DESTROYER,
            Card::BattleBlob => &BATTLE_BLOB,
            Card::BlobCarrier => &BLOB_CARRIER,
            Card::Mothership => &MOTHERSHIP,
            Card::BlobWheel => &BLOB_WHEEL,
            Card::TheHive => &THE_HIVE,
            Card::BlobWorld => &BLOB_WORLD,
            Card::FederationShuttle => &FEDERATION_SHUTTLE,
            Card::Cutter => &CUTTER,
            Card::EmbassyYacht => &EMBASSY_YACHT,
            Card::Freighter => &FREIGHTER,
            Card::TradeEscort => &TRADE_ESCORT,
            Card::Flagship => &FLAGSHIP,
            Card::CommandShip => &COMMAND_SHIP,
            Card::TradingPost => &TRADING_POST,
            Card::BarterWorld => &BARTER_WORLD,
            Card::DefenseCenter => &DEFENSE_CENTER,
            Card::PortOfCall => &PORT_OF_CALL,
            Card::CentralOffice => &CENTRAL_OFFICE,
            Card::ImperialFighter => &IMPERIAL_FIGHTER,
            Card::Corvette => &CORVETTE,
            Card::ImperialFrigate => &IMPERIAL_FRIGATE,
            Card::SurveyShip => &SURVEY_SHIP,
            Card::Battlecruiser => &BATTLECRUISER,
            Card::Dreadnaught => &DREADNAUGHT,
            Card::SpaceStation => &SPACE_STATION,
            Card::RecyclingStation => &RECYCLING_STATION,
            Card::WarWorld => &WAR_WORLD,
            Card::RoyalRedoubt => &ROYAL_REDOUBT,
            Card::FleetHq => &FLEET_HQ,
            Card::TradeBot => &TRADE_BOT,
            Card::MissileBot => &MISSILE_BOT,
            Card::SupplyBot => &SUPPLY_BOT,
            Card::PatrolMech => &PATROL_MECH,
            Card::StealthNeedle => &STEALTH_NEEDLE,
            Card::BattleMech => &BATTLE_MECH,
            Card::MissileMech => &MISSILE_MECH,
            Card::BattleStation => &BATTLE_STATION,
            Card::MechWorld => &MECH_WORLD,
            Card::Junkyard => &JUNKYARD,
            Card::MachineBase => &MACHINE_BASE,
            Card::BrainWorld => &BRAIN_WORLD,
            Card::NeedleAs(i) => &needle_copies()[usize::from(i)],
        }
    }

    pub fn name(self) -> &'static str {
        self.data().name
    }

    /// The card named `name`: one of the catalogue's, or a copy Stealth
    /// Needle makes (`Stealth Needle (Ram)`).
    pub fn from_name(name: &str) -> Option<Card> {
        if let Some(card) = Card::ALL.into_iter().find(|card| card.name() == name) {
            return Some(card);
        }
        let target = name
            .strip_prefix(Card::StealthNeedle.name())?
            .strip_prefix(" (")?
            .strip_suffix(')')?;
        let target = Card::from_name(target)?;
        Card::copy_target(target).then(|| Card::needle_as(target))
    }

    /// Whether Stealth Needle can copy `target` (`rules._copy_targets`): a
    /// ship of the catalogue's, other than a Needle.
    pub fn copy_target(target: Card) -> bool {
        target.is_ship() && !target.copies_ships() && target.copied().is_none()
    }

    /// Stealth Needle as a copy of `target`, a ship of the catalogue's
    /// (`rules._copy`).
    pub fn needle_as(target: Card) -> Card {
        let i = Card::ALL
            .iter()
            .position(|&c| c == target)
            .expect("a catalogue card");
        Card::NeedleAs(u8::try_from(i).expect("fewer than 256 cards"))
    }

    /// The ship this card copies, if it is a copy Stealth Needle made.
    pub fn copied(self) -> Option<Card> {
        match self {
            Card::NeedleAs(i) => Some(Card::ALL[usize::from(i)]),
            _ => None,
        }
    }

    /// The card this is in every pile but in play (`rules.original`):
    /// Stealth Needle for a copy it made, else itself.
    pub fn original(self) -> Card {
        match self {
            Card::NeedleAs(_) => Card::StealthNeedle,
            _ => self,
        }
    }

    pub fn is_ship(self) -> bool {
        self.data().card_type == CardType::Ship
    }

    pub fn is_outpost(self) -> bool {
        self.data().card_type == CardType::Outpost
    }

    /// Whether the card's primary ability copies a ship (Stealth Needle).
    pub fn copies_ships(self) -> bool {
        self.data().primary.contains(&CopyShip)
    }

    /// Whether this card, in play, is an ally for `faction`
    /// (`rules._allied`): a card of that faction, or one that is an ally for
    /// other factions too (`rules._OTHER_FACTIONS`): Mech World, for every
    /// faction, and a copy Stealth Needle made of a ship of another faction,
    /// for the Needle's own.
    pub fn ally_for(self, faction: Faction) -> bool {
        let data = self.data();
        if data.faction == Some(faction) {
            return true;
        }
        if data.primary.contains(&AllyForAllFactions) {
            return true;
        }
        match self.copied() {
            Some(_) => {
                let needle = Card::StealthNeedle.data().faction;
                needle == Some(faction) && data.faction != needle
            }
            None => false,
        }
    }

    /// The combat each ship played gains while this card is in play
    /// (`rules._ship_bonus`, Fleet HQ).
    pub fn ship_bonus(self) -> i32 {
        self.data()
            .primary
            .iter()
            .map(|e| match e {
                ShipsGainCombat(amount) => *amount,
                _ => 0,
            })
            .sum()
    }

    /// The options of the card's "or" choice, its one `Choice` effect
    /// (`rules._options`).
    pub fn options(self) -> &'static [&'static [Effect]] {
        let data = self.data();
        data.primary
            .iter()
            .chain(data.ally)
            .chain(data.scrap)
            .find_map(|e| match e {
                Choice(options) => Some(*options),
                _ => None,
            })
            .expect("the card has an \"or\" choice")
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

/// The Trade Deck: every faction card, each repeated `copies` times, in
/// catalogue order (`catalogue.TRADE_DECK`), so a seed shuffles it as Python
/// does.
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
    fn the_trade_deck_has_80_cards_20_per_faction() {
        let deck = trade_deck();
        assert_eq!(deck.len(), 80);
        for faction in Faction::ALL {
            let n = deck
                .iter()
                .filter(|card| card.data().faction == Some(faction))
                .count();
            assert_eq!(n, 20, "{faction:?}");
        }
    }

    #[test]
    fn names_round_trip() {
        for card in Card::ALL {
            assert_eq!(Card::from_name(card.name()), Some(card));
        }
        let copy = Card::needle_as(Card::Ram);
        assert_eq!(copy.name(), "Stealth Needle (Ram)");
        assert_eq!(Card::from_name("Stealth Needle (Ram)"), Some(copy));
        assert_eq!(Card::from_name("Stealth Needle (Blob Wheel)"), None);
        assert_eq!(Card::from_name("Nebula"), None);
    }

    #[test]
    fn a_copy_has_its_target_s_abilities_and_faction() {
        let copy = Card::needle_as(Card::Ram);
        assert_eq!(copy.data().faction, Some(Faction::Blob));
        assert_eq!(copy.data().cost, 4);
        assert!(copy.ally_for(Faction::Blob));
        assert!(copy.ally_for(Faction::MachineCult));
        assert!(!copy.ally_for(Faction::StarEmpire));
        assert_eq!(copy.original(), Card::StealthNeedle);
        let scout = Card::needle_as(Card::Scout);
        assert_eq!(scout.data().faction, Some(Faction::MachineCult));
    }
}
